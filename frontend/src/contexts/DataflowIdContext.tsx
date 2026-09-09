import { createContext, useContext } from "react";

// The current dataflow's id, threaded down to any node that needs to tell
// the backend which dataflow it's running inside of - e.g. a pyCodeEditorNode
// sending `dataflow_id` to /api/run-python so a Compute Catalog model's bare
// relative paths ("computed/A.geojson") resolve to *this* dataflow's own
// computed dir (see _compute_scratch_dir in backend/server.py). Provided once
// in App.tsx, where the id is already in scope for save/load.
const DataflowIdContext = createContext<string | null>(null);

export const DataflowIdProvider = DataflowIdContext.Provider;

export function useDataflowId(): string | null {
  return useContext(DataflowIdContext);
}
