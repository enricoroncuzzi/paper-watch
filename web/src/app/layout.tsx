import type { Metadata } from "next";
import { cookies } from "next/headers";
import { Inter, JetBrains_Mono, Newsreader } from "next/font/google";

import "./globals.css";

// Self-hosted by Next at build time, exposed as the three variables globals.css reads.
const newsreader = Newsreader({
  subsets: ["latin"],
  style: ["normal", "italic"],
  axes: ["opsz"],
  variable: "--font-newsreader",
  display: "swap",
});
const inter = Inter({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-inter",
  display: "swap",
});
const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  weight: ["400"],
  variable: "--font-jetbrains-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Paper Watch",
  description: "A private reading room for the AI-research corpus.",
};

export default async function RootLayout({ children }: { children: React.ReactNode }) {
  // The theme is a cookie so the server renders the right palette: a client-side toggle would
  // flash the other theme on every reload.
  const theme = (await cookies()).get("pw-theme")?.value;
  const dataTheme = theme === "light" || theme === "dark" ? theme : undefined;
  return (
    <html
      lang="en"
      data-theme={dataTheme}
      className={`${newsreader.variable} ${inter.variable} ${jetbrainsMono.variable}`}
    >
      <body>{children}</body>
    </html>
  );
}
