import { pdfjs } from "react-pdf"
import { appPath } from "@/lib/paths"

// next.config.mjs copies the worker into public/. Import this module before rendering any react-pdf `Document`
pdfjs.GlobalWorkerOptions.workerSrc = appPath("/pdf.worker.min.mjs")
