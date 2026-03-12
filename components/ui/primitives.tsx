import * as React from 'react';
import { cn } from '@/lib/utils';

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'default' | 'ghost' | 'outline' | 'destructive' | 'secondary';
  size?: 'default' | 'sm' | 'lg' | 'icon';
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = 'default', size = 'default', ...props }, ref) => {
    const variants: Record<string, string> = {
      default: 'bg-primary text-primary-foreground hover:bg-primary/90',
      ghost: 'hover:bg-accent hover:text-accent-foreground',
      outline: 'border border-border bg-transparent hover:bg-accent',
      destructive: 'bg-destructive text-destructive-foreground hover:bg-destructive/90',
      secondary: 'bg-muted text-foreground hover:bg-muted/80',
    };
    const sizes: Record<string, string> = {
      default: 'h-9 px-4 py-2 text-sm',
      sm: 'h-8 px-3 text-xs',
      lg: 'h-10 px-6 text-sm',
      icon: 'h-8 w-8',
    };
    return (
      <button
        ref={ref}
        className={cn(
          'inline-flex items-center justify-center gap-2 rounded-md font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50',
          variants[variant], sizes[size], className
        )}
        {...props}
      />
    );
  }
);
Button.displayName = 'Button';

export const Input = React.forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(
  ({ className, ...props }, ref) => (
    <input
      ref={ref}
      className={cn(
        'flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm transition-colors placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
        className
      )}
      {...props}
    />
  )
);
Input.displayName = 'Input';

export function Badge({ className, variant = 'default', children }: {
  className?: string;
  variant?: 'default' | 'secondary' | 'outline';
  children: React.ReactNode;
}) {
  const variants: Record<string, string> = {
    default: 'bg-primary/15 text-primary border-primary/20',
    secondary: 'bg-muted text-muted-foreground border-muted',
    outline: 'bg-transparent text-muted-foreground border-border',
  };
  return (
    <span className={cn(
      'inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] font-medium transition-colors',
      variants[variant], className
    )}>
      {children}
    </span>
  );
}

export function ScrollArea({ className, children }: { className?: string; children: React.ReactNode }) {
  return (
    <div className={cn('overflow-y-auto overflow-x-hidden', className)}>
      {children}
    </div>
  );
}

export function Separator({ className, orientation = 'horizontal' }: {
  className?: string;
  orientation?: 'horizontal' | 'vertical';
}) {
  return (
    <div
      className={cn(
        'shrink-0 bg-border',
        orientation === 'horizontal' ? 'h-px w-full' : 'h-full w-px',
        className
      )}
    />
  );
}

export function Dialog({ open, onClose, children }: {
  open: boolean;
  onClose: () => void;
  children: React.ReactNode;
}) {
  const [mounted, setMounted] = React.useState(false);
  const [visible, setVisible] = React.useState(false);

  React.useEffect(() => {
    if (open) {
      setMounted(true);
      requestAnimationFrame(() => {
        requestAnimationFrame(() => setVisible(true));
      });
    } else {
      setVisible(false);
      const timer = setTimeout(() => setMounted(false), 200);
      return () => clearTimeout(timer);
    }
  }, [open]);

  if (!mounted) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-[10vh] md:pt-[15vh] px-3 md:px-0">
      <div
        className={cn(
          'fixed inset-0 bg-black/60 backdrop-blur-sm transition-opacity duration-200 ease-out',
          visible ? 'opacity-100' : 'opacity-0'
        )}
        onClick={onClose}
      />
      <div className={cn(
        'relative z-10 w-full max-w-lg transition-all duration-200 ease-out',
        visible ? 'opacity-100 scale-100 translate-y-0' : 'opacity-0 scale-95 -translate-y-3'
      )}>
        {children}
      </div>
    </div>
  );
}

export function DialogContent({ className, children }: { className?: string; children: React.ReactNode }) {
  return (
    <div className={cn(
      'rounded-xl border border-border bg-card p-0 shadow-2xl shadow-black/20',
      className
    )}>
      {children}
    </div>
  );
}

export function Tooltip({ children, content, side = 'top' }: { children: React.ReactNode; content: string; side?: 'top' | 'right' | 'bottom' | 'left' }) {
  const positions: Record<string, string> = {
    top: 'bottom-full left-1/2 -translate-x-1/2 mb-1.5',
    right: 'left-full top-1/2 -translate-y-1/2 ml-1.5',
    bottom: 'top-full left-1/2 -translate-x-1/2 mt-1.5',
    left: 'right-full top-1/2 -translate-y-1/2 mr-1.5',
  };
  return (
    <span className="relative group inline-flex">
      {children}
      <span className={cn(
        'absolute px-2 py-1 text-xs bg-foreground text-background rounded opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap z-50',
        positions[side]
      )}>
        {content}
      </span>
    </span>
  );
}
