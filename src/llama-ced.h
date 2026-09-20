#pragma once

// DeepSeek-V4.1 CED bounded replay: which rows of a ubatch run through the decoder.
//
// Under CED the decoder's global KV is projected from the encoder output, so a prompt token only needs
// the decoder for its own window KV and its logits. The reference serving replays the last n_win prompt
// tokens through the decoder and stops everything else at the encoder (tech report 2.2 / 3.2.2). Here a
// row is kept when its position is at or past the sequence's replay start, or when it is an output row.
// Shared by the graph builder (which subsets the activations at the decoder boundary), the per-ubatch
// input (which fills the row list) and the context (which scatters per-layer extractions back to batch
// rows), so all three agree on the same set.

#include "llama-batch.h"
#include "llama-cparams.h"

#include <vector>

// Fills `rows` with the ubatch indices that run through the decoder. Returns false when bounded replay is
// off or every row qualifies (the caller then keeps the unchanged full-batch path); in that case `rows` is
// left empty.
static inline bool llama_ced_dec_rows(const llama_cparams & cparams, const llama_ubatch & ubatch, std::vector<int32_t> & rows) {
    rows.clear();
    if (cparams.ced_replay_window == 0 || ubatch.pos == nullptr || ubatch.seq_id == nullptr) {
        return false;
    }
    bool any_skipped = false;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        const llama_seq_id seq  = ubatch.n_seq_id && ubatch.n_seq_id[i] > 0 ? ubatch.seq_id[i][0] : -1;
        const llama_pos    from = (seq >= 0 && (size_t) seq < cparams.ced_replay_from.size()) ? cparams.ced_replay_from[seq] : -1;
        const bool         out  = ubatch.output && ubatch.output[i] != 0;
        if (from < 0 || ubatch.pos[i] >= from || out) {
            rows.push_back((int32_t) i);
        } else {
            any_skipped = true;
        }
    }
    if (!any_skipped) {
        rows.clear();
        return false;
    }
    return true;
}
