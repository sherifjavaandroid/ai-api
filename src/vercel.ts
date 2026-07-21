"use strict";

// Vercel serverless entry point.
// Unlike src/index.ts (which starts a long-lived Koa server via app.listen),
// this module builds the app once per cold start and exports the request
// handler so Vercel can invoke it as a serverless function.

import "@/lib/initialize.ts";
import server from "@/lib/server.ts";
import routes from "@/api/routes/index.ts";

server.attachRoutes(routes);

// Koa's callback is a standard (req, res) Node handler that @vercel/node accepts.
export default server.app.callback();
