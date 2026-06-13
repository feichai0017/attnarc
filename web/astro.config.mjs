// @ts-check
import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";

// QuillCache docs site — Astro + Starlight, Claude-themed.
// Deployed to GitHub Pages at https://feichai0017.github.io/quillcache/
const SITE_BASE = "/quillcache";

// Astro/Starlight base-prefixes the *sidebar* but NOT root-absolute links written
// in markdown/MDX prose (`[x](/storage-study/)`), so those 404 under a project
// base. This rehype plugin rewrites every root-absolute internal <a href> to
// include the base, once, centrally — so prose links can stay base-agnostic.
function rehypeBaseUrl() {
  const walk = (node) => {
    if (
      node.tagName === "a" &&
      node.properties &&
      typeof node.properties.href === "string"
    ) {
      const href = node.properties.href;
      if (
        href.startsWith("/") &&
        !href.startsWith("//") &&
        !href.startsWith(`${SITE_BASE}/`) &&
        href !== SITE_BASE
      ) {
        node.properties.href = SITE_BASE + href;
      }
    }
    if (Array.isArray(node.children)) node.children.forEach(walk);
  };
  return (tree) => walk(tree);
}

export default defineConfig({
  site: "https://feichai0017.github.io",
  base: SITE_BASE,
  markdown: {
    rehypePlugins: [rehypeBaseUrl],
  },
  integrations: [
    starlight({
      title: "QuillCache",
      description:
        "A Mooncake/Dynamo-style distributed KV cache pool and control plane for LLM serving, in Rust — with identity-governed safe reuse and a crash-consistent persistent tier.",
      social: {
        github: "https://github.com/feichai0017/quillcache",
      },
      customCss: ["./src/styles/claude.css"],
      head: [
        {
          tag: "link",
          attrs: { rel: "preconnect", href: "https://fonts.googleapis.com" },
        },
        {
          tag: "link",
          attrs: {
            rel: "preconnect",
            href: "https://fonts.gstatic.com",
            crossorigin: true,
          },
        },
        {
          tag: "link",
          attrs: {
            rel: "stylesheet",
            href: "https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=Inter:wght@400;450;500;600&display=swap",
          },
        },
      ],
      editLink: {
        baseUrl: "https://github.com/feichai0017/quillcache/edit/main/web/",
      },
      lastUpdated: true,
      sidebar: [
        {
          label: "Start here",
          items: [
            { label: "Overview", link: "/overview/" },
            { label: "Quick start", link: "/quickstart/" },
          ],
        },
        {
          label: "Architecture",
          items: [
            { label: "How it fits together", link: "/architecture/" },
            { label: "Crates", link: "/crates/" },
            { label: "Mooncake / Dynamo mapping", link: "/reference-mapping/" },
          ],
        },
        {
          label: "Deep dives",
          items: [
            { label: "ART vs LSM storage study", link: "/storage-study/" },
            { label: "Identity-safe reuse", link: "/identity-safe-reuse/" },
            { label: "Crash-consistent tier", link: "/crash-consistency/" },
          ],
        },
      ],
    }),
  ],
});
