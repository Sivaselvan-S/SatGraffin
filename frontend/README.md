# SatGraffin Copilot UI

An elegant React + TypeScript frontend for the SatGraffin retrieval-augmented chatbot. It connects to the FastAPI backend under `../backend` and surfaces grounded answers enriched with citation links.

## ✨ Features

- Conversational interface with animated message bubbles and chat history persistence.
- Calls the `/api/query` endpoint and renders `source_links` as rich citation pills.
- Status banner, loading indicators, and graceful error fallback messaging.
- Local storage persistence so a page refresh keeps your last conversation.
- Responsive glassmorphism styling with keyboard-friendly controls.

## 🚀 Getting started

```powershell
cd ..\frontend
npm install
npm run dev
```

The app expects the backend running at `http://localhost:8000` by default. To target a different URL, create a `.env` file in this folder and set:

```env
VITE_API_BASE_URL=https://your-backend-hostname
```

## 📁 Project structure

```
frontend/
├── index.html             # Document shell + fonts
├── package.json
├── public/
│   └── satgraffin.svg     # Favicon
├── src/
│   ├── App.tsx            # Main application shell
│   ├── components/        # Header, chat UI, status, etc.
│   ├── hooks/             # Local storage chat history helper
│   ├── styles/            # Global + component styles
│   └── types.ts           # Shared TypeScript contracts
└── vite.config.ts
```

## 🧪 Useful scripts

```powershell
npm run dev       # Start Vite dev server (http://localhost:5173)
npm run build     # Type-check and create a production bundle in dist/
npm run preview   # Preview the production build locally
npm run lint      # ESLint checks (optional)
```

Before deploying, ensure the FastAPI backend exposes `/api/query` with the response shape:

```ts
interface QueryResponse {
  response: string
  source_links?: string[]
}
```

## 🧭 Next steps

- Wire the UI to user authentication if the backend introduces tokens.
- Stream responses for longer generations using Server-Sent Events.
- Add skeleton responses for yet-to-be-implemented backend features.
