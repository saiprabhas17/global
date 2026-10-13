import logging
import uuid
from datetime import UTC, datetime, timedelta

from datasketch import MinHash, MinHashLSH
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.article import Article

logger = logging.getLogger(__name__)

# Number of permutations for MinHash
NUM_PERMUTATIONS = 128


class NearDuplicateDetector:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.settings = get_settings()
        self.threshold = getattr(self.settings, 'NEAR_DUP_JACCARD_THRESHOLD', 0.75)
        self.lsh = MinHashLSH(threshold=self.threshold, num_perm=NUM_PERMUTATIONS)
        self.article_map: dict[str, Article] = {}
        self.is_initialized = False
        
    def _shingle(self, text: str) -> list[str]:
        """Create 3-gram shingles for text."""
        words = text.casefold().split()
        # Empty MinHashes are identical; indexing short articles with them
        # incorrectly groups every unrelated one/two-word article together.
        if len(words) < 3:
            return [" ".join(words)] if words else []
        return [" ".join(words[i:i+3]) for i in range(len(words) - 2)]
        
    def _create_minhash(self, text: str) -> MinHash:
        m = MinHash(num_perm=NUM_PERMUTATIONS)
        for shingle in self._shingle(text):
            m.update(shingle.encode('utf8'))
        return m

    async def initialize(self) -> None:
        """Loads articles from the bounded window into the LSH index."""
        window_days = getattr(self.settings, 'NEAR_DUP_WINDOW_DAYS', 14)
        cutoff_date = datetime.now(UTC) - timedelta(days=window_days)
        
        # Only the columns the index needs; rows become transient snapshots
        # (never session-bound) so a later rollback cannot expire them.
        stmt = select(
            Article.id, Article.cleaned_content, Article.collected_at, Article.canonical_article_id
        ).where(
            Article.collected_at >= cutoff_date,
            Article.cleaned_content.is_not(None)
        )
        result = await self.db.execute(stmt)

        for row in result.all():
            if not row.cleaned_content:
                continue
            article = Article(
                id=row.id,
                cleaned_content=row.cleaned_content,
                collected_at=row.collected_at,
                canonical_article_id=row.canonical_article_id,
            )
            m = self._create_minhash(row.cleaned_content)
            # Use string representation of UUID as key
            key = str(article.id)
            self.lsh.insert(key, m)
            self.article_map[key] = article
            
        self.is_initialized = True
        logger.info(f"Initialized near-duplicate detector with {len(self.article_map)} articles")

    def find_and_insert(self, new_article: Article) -> uuid.UUID | None:
        """
        Finds if the new_article is a near duplicate of existing ones.
        If it is, returns the canonical_article_id it should link to.
        Also inserts this new_article into the LSH index for subsequent comparisons.
        """
        if not self.is_initialized:
            raise RuntimeError("NearDuplicateDetector must be initialized first")
            
        if not new_article.cleaned_content:
            return None
            
        m = self._create_minhash(new_article.cleaned_content)
        
        # Query LSH
        result_keys = self.lsh.query(m)
        
        canonical_id = None
        
        if result_keys:
            # We found near-duplicates!
            # Find the earliest collected article in this cluster.
            earliest_article = None
            for key in result_keys:
                match = self.article_map.get(key)
                if not match or match.id == new_article.id or not match.cleaned_content:
                    continue
                # LSH only generates candidates. Verify actual Jaccard before
                # collapsing evidence provenance into one canonical source.
                left = set(self._shingle(new_article.cleaned_content))
                right = set(self._shingle(match.cleaned_content))
                union = left | right
                if not union or len(left & right) / len(union) < self.threshold:
                    continue
                # If the matched article is already a duplicate of something else,
                # its canonical root is definitely older than this match.
                if not earliest_article or self._sort_key(match) < self._sort_key(earliest_article):
                    earliest_article = match
            
            if earliest_article:
                canonical_id = earliest_article.canonical_article_id or earliest_article.id
                
        # Insert the new article into the index so it can match against subsequent articles in this run
        new_key = str(new_article.id)
        if new_key in self.article_map:
            self.lsh.remove(new_key)
        self.lsh.insert(new_key, m)
        self.article_map[new_key] = new_article
        
        return canonical_id

    @staticmethod
    def _sort_key(article: Article) -> tuple[datetime, str]:
        observed = article.collected_at or datetime.max.replace(tzinfo=UTC)
        if observed.tzinfo is None:
            observed = observed.replace(tzinfo=UTC)
        return observed, str(article.id)
