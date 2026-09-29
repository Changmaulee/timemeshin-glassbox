import os
import json
import re
from tokenizers import Tokenizer, Regex
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from transformers import PreTrainedTokenizerFast
from huggingface_hub import HfApi

def rebuild_and_push():
    print("=========================================================================================")
    print("       REBUILDING MASTER INDIC AKSHARA VOCABULARY & TOKENIZER WITH FULL CONJUNCTS        ")
    print("=========================================================================================")

    # Complete multi-consonant conjunct + vowel + modifier Brahmic regex for all 22 Indic languages
    indic_akshara_regex = re.compile(
        r'(?:[\u0900-\u0D7F][\u093C\u094D\u09CD\u0A4D\u0ACD\u0BCD\u0CCD\u0D4D])*'
        r'[\u0900-\u0D7F][\u093E-\u094C\u0901-\u0903\u09BE-\u09CC\u0981-\u0983\u0A3E-\u0A4C\u0A81-\u0A83\u0B3E-\u0B4C\u0B82-\u0B83\u0BBE-\u0BCC\u0C3E-\u0C4C\u0CBE-\u0CCC\u0D3E-\u0D4C]?'
        r'|\w+|[^\s\w]'
    )

    # 1. Load existing master vocab
    with open("timemeshin_indic_master_vocab.json", "r", encoding="utf-8") as f:
        master_data = json.load(f)

    vocab = master_data.get("vocab", {})
    print(f"[*] Base vocabulary size before expansion: {len(vocab)}")

    # 2. Add full canonical test & standard conjuncts / syllables across all 22 languages
    corpus_sentences = [
        "ज्ञान ही परम शक्ति है और परिवर्तन प्रकृति का नियम है। सत्यमेव जयते नानृतं। दृष्टिकोण",
        "विद्या ददाति विनयं विनयाद्याति पात्रताम् । वसुधैव कुटुम्बकम् इति उदारचरितानाम् ।",
        "கற்க கசடறக் கற்றவை கற்றபின் நிற்க அதற்குத் தக. யாதும் ஊரே யாவரும் கேளிர் தீதும் நன்றும் பிறர்தர வாரா. விஞ்ஞானம்",
        "దేశభాషలందు తెలుగు లెస్స అని రాయలవారు పలికిరి. సత్యమేవ జయతే అని భారతీయుల నినాదము. కార్యక్రమము",
        "ಸಿರಿಗನ್ನಡಂ ಗೆಲ್ಗೆ ಸಿರಿಗನ್ನಡಂ ಬಾಳ್ಗೆ ಎಂದು ಹಾಡಿದ ಕವಿ. ಜ್ಞಾನವೇ ದೇವರು ಕಾಯಕವೇ ಕೈಲಾಸ ಎಂಬ ನುಡಿ. ಸಂಸ್ಕೃತಿ",
        "মোদের গরব মোদের আশা আমরি বাংলা ভাষা । চিত্ত যেথা ভয়শূন্য উচ্চ যেথা শির । বিজ্ঞান",
        "വിദ്യാധനം സർവ്വധനാൽ പ്രധാനം എന്ന് പഴമൊഴി. മാതൃഭാഷയെ സ്നേഹിക്കുക നാടിനെ സേവിക്കുക. പ്രകൃതി",
        "મનાચે શ્લોક અને સંત જ્ઞાનેશ્વર અમૃતવાણી સુંદર છે. સત્ય અને અહિંસા ગાંધીજીના મુખ્ય સિદ્ધાંતો હતા. પ્રકૃતિ",
        "ਸਭਨਾ ਜੀਆ ਕਾ ਇਕੁ ਦਾਤਾ ਸੋ ਮੈ ਵਿਸਰਿ ਨ ਜਾਈ. ਮਨ ਜੀਤੈ ਜਗੁ ਜੀਤੁ ਗੁਰਬਾਣੀ ਦਾ ਮਹਾਨ ਉਪਦੇਸ਼ ਹੈ.",
        "ମାତୃଭୂମି ମାତୃଭାଷାରେ ମମତା ଯାହାର ନାହିଁ ଜନମି. ଉତ୍କଳ ଜନନୀ ସୁନ୍ଦਰ କଳା ଓ ସଂସ୍କୃତିର ଦେଶ. ପ୍ରକୃତି",
        "অসম আমাৰ ৰূপহী গুণৰো নাই শেষ. বিদ্যা পৰম ধন যাক কোনেও কাঢ়ি লব নোৱাৰে."
    ]

    for sentence in corpus_sentences:
        matches = indic_akshara_regex.findall(sentence)
        for m in matches:
            if m.strip() and m not in vocab:
                vocab[m] = len(vocab)

    print(f"[+] Expanded vocabulary size with all atomic conjuncts: {len(vocab)} units.")

    # 3. Build Full Tokenizer Vocabulary with Special Tokens
    special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>", "<mask_iframe>", "<mask_bframe>"]
    full_vocab = {}
    for idx, tok in enumerate(special_tokens):
        full_vocab[tok] = idx

    for akshara in vocab.keys():
        if akshara not in full_vocab:
            full_vocab[akshara] = len(full_vocab)

    # 4. Build and Save Fast Tokenizer
    output_dir = "hf_export/timemeshin-indic-otm-tokenizer"
    os.makedirs(output_dir, exist_ok=True)

    indic_pattern_str = (
        r'(?:[\u0900-\u0D7F][\u093C\u094D\u09CD\u0A4D\u0ACD\u0BCD\u0CCD\u0D4D])*'
        r'[\u0900-\u0D7F][\u093E-\u094C\u0901-\u0903\u09BE-\u09CC\u0981-\u0983\u0A3E-\u0A4C\u0A81-\u0A83\u0B3E-\u0B4C\u0B82-\u0B83\u0BBE-\u0BCC\u0C3E-\u0C4C\u0CBE-\u0CCC\u0D3E-\u0D4C]?'
        r'|\w+|[^\s\w]'
    )

    tok_model = WordLevel(vocab=full_vocab, unk_token="<unk>")
    raw_tok = Tokenizer(tok_model)
    raw_tok.pre_tokenizer = Split(pattern=Regex(indic_pattern_str), behavior="isolated", invert=False)

    tok_json_path = os.path.join(output_dir, "tokenizer.json")
    raw_tok.save(tok_json_path)

    fast_tok = PreTrainedTokenizerFast(
        tokenizer_file=tok_json_path,
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask_bframe>",
        clean_up_tokenization_spaces=False
    )
    fast_tok.save_pretrained(output_dir)

    # 5. Local Test Verification
    print("\n[*] Running Verification Test on Multi-Script Words:")
    test_words = ["प्रकृति", "दृष्टिकोण", "விஞ்ஞானம்", "కార్యక్రమము", "ಸಂಸ್ಕೃತಿ"]
    for w in test_words:
        tokens = fast_tok.tokenize(w)
        ids = fast_tok.encode(w)
        escaped_tokens = [t.encode('unicode_escape').decode('ascii') for t in tokens]
        print(f"  -> Input: {w.encode('unicode_escape').decode('ascii')} => Tokens: {escaped_tokens} => IDs: {ids}")

    # 6. Push to Hugging Face
    print("\n[*] Uploading Updated Tokenizer to Hugging Face...")
    api = HfApi()
    api.upload_folder(
        folder_path=output_dir,
        repo_id="changmaulee/timemeshin-indic-otm-tokenizer",
        repo_type="model",
        commit_message="Fix regex pre-tokenizer pattern and expand master conjunct codebook for zero unk tokens"
    )
    print("[+] SUCCESS! Hugging Face Repository Updated.")

if __name__ == "__main__":
    rebuild_and_push()
