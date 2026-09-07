export default function Logo({ size = 22 }: { size?: number }) {
  return (
    <svg
      className="brand-mark"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
      style={{ width: size, height: size }}
    >
      <path d="M12 3.2 L20.2 7.6 L20.2 16.4 L12 20.8 L3.8 16.4 L3.8 7.6 Z" stroke="currentColor" strokeOpacity="0.35" strokeWidth="1.2" />
      <circle cx="12" cy="12" r="3" fill="#5b7cfa" />
      <circle cx="12" cy="3.2" r="1.6" fill="currentColor" fillOpacity="0.9" />
      <circle cx="20.2" cy="7.6" r="1.6" fill="currentColor" fillOpacity="0.75" />
      <circle cx="20.2" cy="16.4" r="1.6" fill="currentColor" fillOpacity="0.6" />
      <circle cx="12" cy="20.8" r="1.6" fill="currentColor" fillOpacity="0.75" />
      <circle cx="3.8" cy="16.4" r="1.6" fill="currentColor" fillOpacity="0.6" />
      <circle cx="3.8" cy="7.6" r="1.6" fill="currentColor" fillOpacity="0.9" />
      <g stroke="#5b7cfa" strokeOpacity="0.55" strokeWidth="1">
        <path d="M12 12 L12 3.2" />
        <path d="M12 12 L20.2 7.6" />
        <path d="M12 12 L3.8 7.6" />
      </g>
    </svg>
  );
}
