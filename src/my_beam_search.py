import math
import heapq
import torch

def ctc_beam_search_fsa(
    log_probs: torch.Tensor,         # [T, C] — лог-софтмакс по классам
    beam_width: int = 5,
    blank: int = 0
) -> list[tuple[list[int], float]]:
    """
    Beam-search с FSA-ограничениями: максимум 1 '|', ≤3 токенов до/после.
    Возвращает до beam_width пар (seq, score), где seq — список индексов.
    """
    T, C = log_probs.shape
    # beam: (score, seq, last, state, thou_cnt, rem_cnt)
    beam = [(0.0, (), None, 0, 0, 0)]

    for t in range(T):
        new_beams = {}
        for score, seq, last, state, thou_cnt, rem_cnt in beam:
            for c in range(C):
                p = log_probs[t, c].item()
                ns = score + p

                # blank всегда разрешён
                if c == blank:
                    key = (seq, last, state, thou_cnt, rem_cnt)
                    new_beams[key] = max(new_beams.get(key, -1e9), ns)
                    continue

                # CTC-коллапс: не повторяем один и тот же токен подряд
                if c == last:
                    continue

                # state: 0=S0 (до '|'), 1=S1 (после '|'), 2=S2 (только blank)
                if state == 0:
                    # до '|': можно брать до 3 любых non-blank
                    if len(seq) < 3:
                        key = (seq + (c,), c, 0, thou_cnt + 1, rem_cnt)
                        new_beams[key] = max(new_beams.get(key, -1e9), ns)
                elif state == 1:
                    # после '|': до 3 остаточных
                    if rem_cnt < 3:
                        key = (seq + (c,), c, 1, thou_cnt, rem_cnt + 1)
                        new_beams[key] = max(new_beams.get(key, -1e9), ns)
                # state==2: ничего, кроме blank, не разрешено

        # топ-K
        beam = heapq.nlargest(
            beam_width,
            [ (s, seq, last, st, th, re)
              for (seq, last, st, th, re), s in new_beams.items() ],
            key=lambda x: x[0]
        )

    return [(list(seq), score) for score, seq, *_ in beam]


def tokens_to_number_string(tokens: list[str]) -> str:
    """
    Преобразует список токенов ["<600>",...,"|",...] обратно в строку числа "650000".
    Если токен '|' на старте → это эквивалент '<1>|'.
    """
    # найдём разделитель
    if "|" in tokens:
        sep = tokens.index("|")
        thousands = tokens[:sep]
        remainder = tokens[sep+1:]
    else:
        thousands = []
        remainder = tokens

    # если первый токен просто '|' — вставим '<1>' автоматически
    if thousands == [] and tokens and tokens[0] == "|":
        thousands = ["<1>"]

    thou = sum(int(t.strip("<>")) for t in thousands if t != "|")
    rem  = sum(int(t.strip("<>")) for t in remainder if t != "|")
    return str(thou * 1000 + rem)
