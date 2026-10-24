import "next-auth";
import "next-auth/jwt";

declare module "next-auth" {
  interface Session {
    /** The numeric GitHub id, carried in the JWT and checked on every server render. */
    githubId?: string;
  }
}

declare module "next-auth/jwt" {
  interface JWT {
    githubId?: string;
  }
}
