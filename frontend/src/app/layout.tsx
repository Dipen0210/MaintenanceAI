import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MaintenanceAI",
  description: "Equipment condition monitoring and maintenance investigation",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className="antialiased"
      >
        {children}
      </body>
    </html>
  );
}
