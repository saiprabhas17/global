import { useEffect, useRef, type ReactNode } from 'react';
import { createPortal } from 'react-dom';

interface DialogProps {
  labelledBy: string;
  onClose: () => void;
  children: ReactNode;
  className?: string;
  backdropClassName?: string;
}

export function Dialog({ labelledBy, onClose, children, className, backdropClassName }: DialogProps) {
  const ref = useRef<HTMLDivElement>(null);
  const closeRef = useRef(onClose);
  closeRef.current = onClose;

  useEffect(() => {
    const previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const element = ref.current;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    element?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        event.stopPropagation();
        closeRef.current();
      } else if (event.key === 'Tab' && element) {
        const focusable = Array.from(element.querySelectorAll<HTMLElement>(
          'button:not(:disabled), a[href], input:not(:disabled), select:not(:disabled), textarea:not(:disabled), [tabindex="0"]',
        ));
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        if (!first) {
          event.preventDefault();
          element.focus();
        } else if (event.shiftKey && (document.activeElement === first || document.activeElement === element)) {
          event.preventDefault();
          last.focus();
        } else if (!event.shiftKey && (document.activeElement === last || document.activeElement === element)) {
          event.preventDefault();
          first.focus();
        }
      }
    };
    document.addEventListener('keydown', onKey, true);
    return () => {
      document.removeEventListener('keydown', onKey, true);
      document.body.style.overflow = previousOverflow;
      if (previousFocus?.isConnected) previousFocus.focus();
    };
  }, []);

  return createPortal(
    <div
      className={backdropClassName ?? 'fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4'}
      onClick={(event) => { if (event.target === event.currentTarget) onClose(); }}
    >
      <div
        ref={ref}
        role="dialog"
        aria-modal="true"
        aria-labelledby={labelledBy}
        tabIndex={-1}
        className={className ?? 'w-full max-w-xl max-h-[85vh] overflow-y-auto scroll-contained card-brutal-dark p-6 space-y-5 text-hermes-bone'}
      >
        {children}
      </div>
    </div>,
    document.body,
  );
}
