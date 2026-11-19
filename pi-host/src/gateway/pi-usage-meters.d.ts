/** Types for the two `pi-usage-meters` core helpers pi-host uses (the package ships plain JS). */
declare module "pi-usage-meters/extensions/core.js" {
  export type ProviderAuthSource = { modelRegistry: { getProviderAuth(provider: string): Promise<unknown> } };
  /** The provider's OAuth bearer token, undefined when it is not logged in with OAuth. */
  export function tokenFor(ctx: ProviderAuthSource, provider: string): Promise<string | undefined>;
  /** GET JSON with a bearer token; a non-2xx response throws an Error carrying `status`. */
  export function getJson(url: string, token: string, headers?: Record<string, string>, timeoutMs?: number): Promise<unknown>;
}
