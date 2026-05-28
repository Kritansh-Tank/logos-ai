"use client";

import { Church } from "lucide-react";

type Denomination = "protestant" | "catholic" | "orthodox" | "nondenominational";

const OPTIONS: { value: Denomination; label: string; icon: string }[] = [
  { value: "nondenominational", label: "Non-Denom",  icon: "✦" },
  { value: "protestant",        label: "Protestant", icon: "✦" },
  { value: "catholic",          label: "Catholic",   icon: "✦" },
  { value: "orthodox",          label: "Orthodox",   icon: "✦" },
];

interface Props {
  value: Denomination;
  onChange: (d: Denomination) => void;
}

export default function DenominationSelector({ value, onChange }: Props) {
  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 10 }}>
        <Church size={12} color="var(--text-faint)" />
        <p className="sidebar-label" style={{ marginBottom: 0 }}>Christian Tradition</p>
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
        {OPTIONS.map((opt) => (
          <button
            key={opt.value}
            className={`denom-pill${value === opt.value ? " active" : ""}`}
            onClick={() => onChange(opt.value)}
          >
            {opt.label}
          </button>
        ))}
      </div>
    </div>
  );
}
