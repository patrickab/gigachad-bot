"use client";

import { createContext, useContext, useMemo, type ComponentProps } from "react";
import {
  CodeBlock,
  MarkdownRenderer,
  MarkmapDiagram,
  MermaidDiagram,
  type CodeBlockProps,
  type CodeRenderers,
  type ColorScheme,
  type LinkPolicy,
  type MarkdownComponents,
} from "@gigachad/chat-ui";
import { useMarkdownBodyRendering, type MarkdownBodyProps } from "@/components/MarkdownBody";
import { TableChart } from "@/components/TableChart";
import { useTheme } from "@/hooks/useTheme";
import { remarkAgentLinks } from "@/lib/agent-links";
import { remarkInlineCodeFileLinks } from "@/lib/file-links";
import { GithubRepoContext, remarkGithubRefs } from "@/lib/github-refs";

type Props = MarkdownBodyProps & Pick<ComponentProps<typeof MarkdownRenderer>, "compact">;
type RemarkPlugins = NonNullable<ComponentProps<typeof MarkdownRenderer>["remarkPlugins"]>;

// `agent://` handles are opened in-app by upstream's `a` renderer, the sanitizer must keep their href.
// Streamdown's URL hardening would drop relative file paths the `a` renderer opens in the file panel.
const LINK_POLICY: LinkPolicy = { extraHrefProtocols: ["agent"], harden: false };

// Streamdown marks a link still streaming in (`[text](http`) with this href, it is not navigable yet.
const INCOMPLETE_LINK_HREF = "streamdown:incomplete-link";

// omp-web's color scheme, read once per message and handed to its fenced code and diagrams.
const ColorSchemeContext = createContext<ColorScheme>("dark");

function ThemedCodeBlock(props: CodeBlockProps) {
  return <CodeBlock {...props} theme={useContext(ColorSchemeContext)} />;
}

function ThemedMermaid({ code }: { code: string }) {
  return <MermaidDiagram code={code} theme={useContext(ColorSchemeContext)} />;
}

function ThemedMarkmap({ code }: { code: string }) {
  return <MarkmapDiagram code={code} theme={useContext(ColorSchemeContext)} />;
}

// gigachad-bot's own code and diagram renderers, so code and diagrams look as in its chat.
const CODE_RENDERERS: CodeRenderers = { code: ThemedCodeBlock, mermaid: ThemedMermaid, markmap: ThemedMarkmap };

/**
 * Markdown for omp's user and assistant text, rendered by gigachad-bot's shared renderer (`@gigachad/chat-ui`).
 * Tools, thinking, compaction and custom messages keep upstream's `MarkdownBody`.
 *
 * Links and images are upstream's renderers (`useMarkdownBodyRendering`), Streamdown passes them the same hast `node`.
 * Only what Streamdown needs differently is set here:
 * - fenced code, mermaid and markmap are the package's `CodeBlock`, `MermaidDiagram` and `MarkmapDiagram` in omp-web's
 *   color scheme (`useTheme().isDark`), upstream's `code`/`pre` and `MermaidBlock` are dropped
 * - inline code and tables keep the package's elements: upstream's `.markdown-table-wrap` and `.markdown-inline-code`
 *   styling would clash with the package's `markdown.css`. Tables still get upstream's `TableChart` with `autoCharts`
 * - the incomplete-link href renders as text, the link policy keeps `agent://` and relative hrefs
 * - remark plugins are omp's link plugins only, Streamdown brings its own GFM and math
 *
 * The root carries the package's `gc-markdown` class and `data-markdown-root`, never upstream's `.markdown-body`. Its
 * typography is the package's `markdown.css` and `code.css`, imported by `custom/theme/index.ts`, with the role tokens
 * mapped in `custom/theme/chat-ui.css`.
 */
export function EmbeddedMarkdown({ className, compact, ...props }: Props) {
  const { children, isStreaming, cwd, onOpenFile, autoCharts = false } = props;
  const { components: upstream } = useMarkdownBodyRendering(props);
  const githubRepo = useContext(GithubRepoContext);
  const { isDark } = useTheme();

  // GitHub refs run first: `agent://Foo#12` must not become an issue link.
  // Inline-code paths link only where a click can open them.
  const remarkPlugins = useMemo<RemarkPlugins>(() => {
    const plugins: RemarkPlugins = [[remarkGithubRefs, { repo: githubRepo }], remarkAgentLinks];
    return onOpenFile ? [...plugins, [remarkInlineCodeFileLinks, { cwd }]] : plugins;
  }, [githubRepo, onOpenFile, cwd]);

  const components = useMemo<MarkdownComponents>(() => {
    const { a: UpstreamLink, img } = upstream;
    if (typeof UpstreamLink !== "function") throw new Error("upstream MarkdownBody no longer renders links with a component, update EmbeddedMarkdown");
    return {
      a(linkProps) {
        return linkProps.href === INCOMPLETE_LINK_HREF ? <>{linkProps.children}</> : <UpstreamLink {...linkProps} />;
      },
      img,
      inlineCode({ ...codeProps }) {
        // `node` is hast metadata, not a DOM attribute.
        delete codeProps.node;
        return <code {...codeProps} />;
      },
      table({ node, children: tableChildren, ...tableProps }) {
        return (
          <TableChart node={node} enabled={autoCharts && !isStreaming}>
            <div className="gc-markdown-table-wrap">
              <table {...tableProps}>{tableChildren}</table>
            </div>
          </TableChart>
        );
      },
    };
  }, [upstream, autoCharts, isStreaming]);

  return (
    <ColorSchemeContext.Provider value={isDark ? "dark" : "light"}>
      <MarkdownRenderer
        content={children}
        streaming={isStreaming}
        compact={compact}
        className={className}
        codeRenderers={CODE_RENDERERS}
        components={components}
        remarkPlugins={remarkPlugins}
        linkPolicy={LINK_POLICY}
      />
    </ColorSchemeContext.Provider>
  );
}
