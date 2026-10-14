import { esc } from "../ui.js";

export async function notFoundView(ctx, err) {
  ctx.title("Not found");
  ctx.crumbs([["Not found"]]);
  const what = err?.message ? ` (“${esc(err.message)}”)` : "";
  ctx.main.innerHTML = `<div class="empty" style="padding:80px 20px">
    <h1 style="margin-bottom:8px">Page not found</h1>
    <p>We couldn't find that page${what}.</p>
    <p style="margin-top:18px"><a class="btn btn-primary" href="/squad" data-link>Go to squad overview</a>
    <a class="btn" href="/players" data-link>Browse athletes</a></p></div>`;
}
