"""Tensor batching for complete inference decisions."""

def collate_serialized(serialized, pad_token_id, *, max_branch_length=None, pad_to_multiple_of=1):
        """Pad serialized inference branches while preserving every option."""
        import torch

        if not serialized:
            raise ValueError("cannot collate an empty decision batch")
        branches = []
        ptr = [0]
        for encoded in serialized:
            if len(encoded.branches) != len(encoded.option_ids):
                raise ValueError("serializer must return one branch for every option")
            branches.extend(encoded.branches)
            ptr.append(len(branches))
        raw_length = max(len(branch.input_ids) for branch in branches)
        if max_branch_length is not None and raw_length > max_branch_length:
            raise ValueError(f"branch length {raw_length} exceeds {max_branch_length}; truncation is disabled")
        length = ((raw_length + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
        if max_branch_length is not None and length > max_branch_length:
            raise ValueError("padding would exceed max_branch_length")
        n, batch, max_k = len(branches), len(serialized), max(len(ex.option_ids) for ex in serialized)
        input_ids = torch.full((n, length), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((n, length), dtype=torch.bool)
        position_ids = torch.zeros((n, length), dtype=torch.long)
        question_end, option_end = [], []
        for index, branch in enumerate(branches):
            size = len(branch.input_ids)
            if not (0 <= branch.question_end <= branch.option_end < size):
                raise ValueError("serializer returned invalid pooling spans")
            input_ids[index, :size] = torch.tensor(branch.input_ids, dtype=torch.long)
            attention_mask[index, :size] = True
            position_ids[index, :size] = torch.arange(size, dtype=torch.long)
            question_end.append(branch.question_end)
            option_end.append(branch.option_end)
        option_mask = torch.zeros((batch, max_k), dtype=torch.bool)
        for index, encoded in enumerate(serialized):
            option_mask[index, :len(encoded.option_ids)] = True
        return {
            "branch_media": [getattr(ex, "media", None) for ex in serialized for _ in ex.branches],
            "input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids,
            "decision_ptr": torch.tensor(ptr, dtype=torch.long),
            "question_end": torch.tensor(question_end, dtype=torch.long),
            "option_end": torch.tensor(option_end, dtype=torch.long),
            "option_mask": option_mask,
            "decision_ids": [ex.decision_id for ex in serialized],
            "option_ids": [ex.option_ids for ex in serialized],
        }
