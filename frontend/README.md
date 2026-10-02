# Frontend

Run `npm install` and `npm run dev` from this directory. Open http://127.0.0.1:5173/. Build with `npm run build`; preview the generated bundle with `npm run preview`.

The overview imports saved evaluation reports directly from the repository. Live assessments use `/api/status`, `/api/analyze`, and `/api/explain/query`, proxied by Vite to http://127.0.0.1:8000. Start the real Python service with `python run.py` from the project root. That command also serves the compiled frontend at port 8000, so a separate Vite process is only needed during development. The interface checks connection health and reconnects automatically; it never fabricates predictions when the service is unavailable.

The API status must supply the twelve clinical feature definitions, seven modality definitions, model readiness, and loaded model metadata. Numeric measurements are validated against that schema, while the server must validate CSV/image content. Uploads are sent only on explicit assessment submission. No assessment data is persisted in browser storage or URLs.

The React shell composes independently maintained overview, assessment, results, and explainability modules with shared types and tokens. These are modules within one Vite deployment, not independently deployed microfrontends. Vercel's microfrontend routing requires distinct Vercel projects and a deployment group; this repository currently has one frontend application. See https://vercel.com/docs/microfrontends for those deployment prerequisites. No remote projects, deployment group, or hosting resources were created.

Design ownership and behavior are documented in the project-root DESIGN.md and UX-CONTRACT.md. Static premium auditing is scoped to this frontend through premium-ui.json.
