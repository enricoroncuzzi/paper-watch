import NextAuth from "next-auth";
import { NextResponse, type NextRequest } from "next/server";

import { buildAuthConfig } from "@/auth.config";

// The proxy only saves an anonymous page load from rendering; every server component and route
// handler still checks the session itself. API routes are excluded because they answer 401
// themselves, and the sign-in page because it must stay reachable while signed out.
const { auth } = NextAuth(buildAuthConfig(process.env));

export default auth((request: NextRequest & { auth: unknown }) => {
  if (request.auth) return NextResponse.next();
  const signIn = new URL("/signin", request.url);
  return NextResponse.redirect(signIn);
});

export const config = {
  matcher: ["/((?!api|_next/static|_next/image|signin|favicon.ico).*)"],
};
