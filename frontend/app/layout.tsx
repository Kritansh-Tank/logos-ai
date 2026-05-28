import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Logos AI — Christian Scripture Assistant",
  description:
    "A grounded, scripture-aware Christian AI assistant. Ask questions about the Bible, explore theology, and generate Christian imagery — every answer verified against 31,102 KJV verses.",
  keywords: ["Christian AI", "Bible assistant", "scripture", "theology", "RAG", "KJV"],
  openGraph: {
    title: "Logos AI — Christian Scripture Assistant",
    description: "Grounded in God's Word. Powered by AI.",
    type: "website",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700&family=Inter:wght@300;400;500;600;700&family=Lora:ital,wght@0,400;0,500;1,400&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="bg-app">{children}</body>
    </html>
  );
}
