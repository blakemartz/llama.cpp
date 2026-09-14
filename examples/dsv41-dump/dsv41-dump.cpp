// llama-dsv41-dump: run ONE forward on a prompt and write whole tensors whose names match a
// regex to a directory as raw f32 (+ a JSON-lines manifest with the ggml ne). The M4 tool for
// diffing the llama.cpp DeepSeek-V4.1 graph against the CPU reference oracle layer by layer.
//
//   DSV41_DUMP_DIR=/path/out  DSV41_DUMP_FILTER='^(hc_init|l_last-[0-9]+|hc_out|result_norm|result_output)$' \
//   llama-dsv41-dump -m model.gguf -p "prompt" [usual llama args]
//
// Env-configured so the common arg parser is untouched. Only contiguous tensors are written
// (the ones named above are); others are reported and skipped.
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <clocale>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <regex>
#include <string>
#include <vector>

struct dump_state {
    std::regex    filter;
    std::string   outdir;
    std::ofstream manifest;
    int           n_written = 0;
    int           n_skipped = 0;
};

static bool dsv41_dump_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (dump_state *) user_data;
    const bool match = std::regex_match(t->name, st->filter);
    if (ask) {
        return match;              // only ask the scheduler to materialise what we will write
    }
    if (!match) {
        return true;
    }
    if (!ggml_is_contiguous(t)) {
        LOG_WRN("dsv41-dump: %s is not contiguous (type %s, op %s) -- skipped\n", t->name, ggml_type_name(t->type), ggml_op_desc(t));
        st->n_skipped++;
        return true;
    }

    const int64_t n = ggml_nelements(t);
    std::vector<uint8_t> raw(ggml_nbytes(t));
    ggml_backend_tensor_get(t, raw.data(), 0, raw.size());

    std::vector<float> f32(n);
    switch (t->type) {
        case GGML_TYPE_F32:  std::memcpy(f32.data(), raw.data(), n*sizeof(float)); break;
        case GGML_TYPE_F16:  for (int64_t i = 0; i < n; ++i) f32[i] = ggml_fp16_to_fp32(((const ggml_fp16_t *) raw.data())[i]); break;
        case GGML_TYPE_BF16: for (int64_t i = 0; i < n; ++i) f32[i] = ggml_bf16_to_fp32(((const ggml_bf16_t *) raw.data())[i]); break;
        // index tensors (ggml_top_k output); exact as f32 below 2^24, which any cache index is
        case GGML_TYPE_I32:  for (int64_t i = 0; i < n; ++i) f32[i] = (float) ((const int32_t *) raw.data())[i]; break;
        default:
            LOG_WRN("dsv41-dump: %s has type %s -- not dumped\n", t->name, ggml_type_name(t->type));
            st->n_skipped++;
            return true;
    }

    std::string fname = std::string(t->name) + ".f32";
    for (auto & c : fname) { if (c == '/' || c == ' ') c = '_'; }
    const std::string path = st->outdir + "/" + fname;
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) {
        LOG_ERR("dsv41-dump: cannot open %s\n", path.c_str());
        return true;
    }
    std::fwrite(f32.data(), sizeof(float), n, f);
    std::fclose(f);

    st->manifest << "{\"name\":\"" << t->name << "\",\"file\":\"" << fname << "\",\"type\":\"" << ggml_type_name(t->type)
                 << "\",\"op\":\"" << ggml_op_desc(t) << "\",\"ne\":[" << t->ne[0] << "," << t->ne[1] << "," << t->ne[2] << "," << t->ne[3] << "]}\n";
    st->manifest.flush();
    st->n_written++;
    LOG_INF("dsv41-dump: wrote %-24s %s ne={%lld,%lld,%lld,%lld}\n", t->name, ggml_type_name(t->type),
            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    dump_state st;
    const char * dir = std::getenv("DSV41_DUMP_DIR");
    const char * flt = std::getenv("DSV41_DUMP_FILTER");
    st.outdir = dir ? dir : "dsv41-dump-out";
    st.filter = std::regex(flt ? flt : "^(hc_init|hc_pre_mix_init|l_last-[0-9]+|hc_out|result_norm|result_output)$");

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    params.cb_eval = dsv41_dump_cb;
    params.cb_eval_user_data = &st;
    params.warmup = false;

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init\n", __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos, true);
    if (tokens.empty()) {
        LOG_ERR("no input tokens\n");
        return 1;
    }
    // the reference encoder always prepends BOS; the GGUF tokenizer metadata may say otherwise
    if (std::getenv("DSV41_FORCE_BOS") && tokens.front() != llama_vocab_bos(vocab)) {
        tokens.insert(tokens.begin(), llama_vocab_bos(vocab));
    }

    std::string mkdir_cmd = "mkdir -p '" + st.outdir + "'";
    if (std::system(mkdir_cmd.c_str()) != 0) { LOG_WRN("mkdir failed for %s\n", st.outdir.c_str()); }
    st.manifest.open(st.outdir + "/manifest.jsonl");
    {
        std::ofstream tf(st.outdir + "/tokens.txt");
        for (auto tok : tokens) { tf << tok << "\n"; }
    }
    LOG_INF("dsv41-dump: %zu tokens (add_bos=%d), dumping to %s with filter %s\n", tokens.size(), (int) add_bos, st.outdir.c_str(), flt ? flt : "(default)");

    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()))) {
        LOG_ERR("decode failed\n");
        return 1;
    }

    // logits of the last token (sanity: the greedy next token)
    const float * logits = llama_get_logits_ith(ctx, -1);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    int best = 0;
    for (int i = 1; i < n_vocab; ++i) { if (logits[i] > logits[best]) best = i; }
    LOG_INF("dsv41-dump: wrote %d tensors, skipped %d; greedy next token id = %d (%s)\n",
            st.n_written, st.n_skipped, best, common_token_to_piece(ctx, best).c_str());
    {
        std::ofstream lf(st.outdir + "/next_token.txt"); lf << best << "\n";
    }

    llama_backend_free();
    return 0;
}
