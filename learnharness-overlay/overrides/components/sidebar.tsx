// No "use client": the root layout, a server component, imports `AppShell` and `SidebarProvider` from here, and a client
// module's `export *` reaches a server component as undefined. Upstream's module stays the client boundary. The
// replacements live in the client module `embed/chrome.tsx`: wrapping upstream here would reach the client as a
// reference to this override and render it twice.
export * from "../../../components/sidebar";

/** Upstream's app frame, its sessions sidebar hidden in drive mode, and its reopen button, hidden and inert then. */
export { AppShellFrame as AppShell, DriveSidebarToggle as SidebarToggle } from "../../embed/chrome";
