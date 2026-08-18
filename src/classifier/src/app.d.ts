// See https://svelte.dev/docs/kit/types#app.d.ts
// for information about these interfaces
declare global {
	namespace App {
		/** Cloudflare bindings and request context exposed to SvelteKit server routes. */
		interface Platform {
			env: Env;
			ctx: ExecutionContext;
			caches: CacheStorage;
			cf?: IncomingRequestCfProperties;
		}

		// interface Error {}
		// interface Locals {}
		// interface PageData {}
		// interface PageState {}
	}

	interface Grecaptcha {
		ready(cb: () => void): void;
		execute(siteKey: string, options: { action: string }): Promise<string>;
	}

	interface Window {
		grecaptcha?: Grecaptcha;
	}
}

export {};
