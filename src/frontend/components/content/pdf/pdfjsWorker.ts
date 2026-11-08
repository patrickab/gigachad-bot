import { pdfjs } from "react-pdf"

// next.config.mjs copies the worker into public/. Import this module before rendering any react-pdf `Document`
pdfjs.GlobalWorkerOptions.workerSrc = "/pdf.worker.min.mjs"
