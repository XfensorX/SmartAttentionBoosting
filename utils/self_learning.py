import torch
import scipy


def get_degree_similarities(
    weights_a: torch.Tensor,
    weights_b: torch.Tensor,
) -> torch.Tensor:
    """
    Returns degree between each individual weight vector (row-vector) of the two input weight matricies

    If the angle is greated than the similarity threshold, it is set to 181 degrees.
    """
    similarities = weights_a @ weights_b.transpose(0, 1)
    assert not (torch.any(similarities > 1.0001) or torch.any(similarities < -1.0001))

    return similarities.clamp(-1.0, 1.0).arccos().rad2deg()


def get_indices_to_match(
    degree_similarities: torch.Tensor, similarity_threshold_in_degree: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a similarity matrix (in degrees), returns the indices of the respective vectors,
    that are the closest to each other and should be matched.

    returns: two tensors with
        - the indices in the 0-dimension
        - the indices of the 1-dimension
    in the order they belong together respectively.
    e.g.:
    (tensor[1,5], tensor [2,4]) -> vectors 1 in 0-dim and 2 in 1-dim should be matched, ...
    """

    OUT_OF_REACH_DEGREE_PENALTY = 99999999999.0

    filtered_degrees = torch.where(
        degree_similarities > similarity_threshold_in_degree,
        torch.full(degree_similarities.shape, OUT_OF_REACH_DEGREE_PENALTY),
        degree_similarities,
    )
    # for better performance, we first permute the matrix to push all non-matching pairs (degree==penalty)
    # to the bottom of the matrix and only calculate based on the rest of the vectors
    # --> 1 if matching, 0 if not matching
    dim0_is_possible_match = (
        (filtered_degrees != OUT_OF_REACH_DEGREE_PENALTY).sum(dim=1).sign()
    )
    dim1_is_possible_match = (
        (filtered_degrees != OUT_OF_REACH_DEGREE_PENALTY).sum(dim=0).sign()
    )

    # Permute possible matched to the upper left corner of the matrix
    dim0_permutation = dim0_is_possible_match.argsort(descending=True)
    dim1_permutation = dim1_is_possible_match.argsort(descending=True)
    permuted_filtered_degrees = filtered_degrees[dim0_permutation, :][
        :, dim1_permutation
    ]
    # all rows and cols without a match should not be included in the assignment search (otherwise they will get matched)
    permuted_idx_dim0, permuted_idx_dim1 = scipy.optimize.linear_sum_assignment(
        permuted_filtered_degrees[
            : sum(dim0_is_possible_match), : sum(dim1_is_possible_match)
        ]
    )
    dim0_idx_to_match = dim0_permutation[permuted_idx_dim0]
    dim1_idx_to_match = dim1_permutation[permuted_idx_dim1]

    # Nevertheless all matches which should have not been merged have to be filtered out
    wrong_match_idx = torch.nonzero(
        filtered_degrees[dim0_idx_to_match, dim1_idx_to_match]
        == OUT_OF_REACH_DEGREE_PENALTY
    ).reshape(-1)

    correct_match_mask = torch.ones(dim0_idx_to_match.shape, dtype=torch.bool)
    correct_match_mask[wrong_match_idx] = False

    return dim0_idx_to_match[correct_match_mask], dim1_idx_to_match[correct_match_mask]


def complement_indices(of: torch.Tensor, indices: torch.Tensor, dim: int = 0):
    all_indices = range(of.shape[dim])
    return torch.LongTensor(list(set(all_indices) - set(indices.tolist())))


def get_new_idx_permutation(
    idx_to_match: torch.Tensor, idx_not_to_match: torch.Tensor, not_to_match_offset: int
):
    length_of_matched = len(idx_to_match)
    total_length = len(idx_to_match) + len(idx_not_to_match)

    new_locations = torch.zeros((total_length,), dtype=torch.long)
    new_locations[idx_to_match] = torch.arange(0, length_of_matched, dtype=torch.long)
    new_locations[idx_not_to_match] = not_to_match_offset + torch.arange(
        length_of_matched, total_length, dtype=torch.long
    )

    return new_locations


def switch_weights_like_previous_layer(
    weights: list[torch.Tensor],
    weight_permutations: dict[int, torch.Tensor],
    this_layer_output_size: int,
    last_output_size: int,
) -> torch.Tensor:
    combined_weights = torch.zeros(
        (
            len(weights),
            this_layer_output_size,
            last_output_size + 1,  # +1 for bias node
        )
    )

    for net_no, net_weight in enumerate(weights):
        combined_weights[
            net_no,
            :,
            torch.cat(
                [torch.tensor([0]), weight_permutations[net_no] + 1]
            ),  # shift for bias node
        ] = net_weight

    return combined_weights


def combine_weights(
    w1: torch.Tensor, w2: torch.Tensor, similarity_threshold_in_degree: float
):

    degree_similarities = get_degree_similarities(w1, w2)

    w1_idx_to_match, w2_idx_to_match = get_indices_to_match(
        degree_similarities, similarity_threshold_in_degree
    )

    w1_idx_not_to_match = complement_indices(w1, w1_idx_to_match)
    w2_idx_not_to_match = complement_indices(w2, w2_idx_to_match)

    new_w = torch.cat(
        [
            (w1[w1_idx_to_match] + w2[w2_idx_to_match]) / 2,
            w1[w1_idx_not_to_match],
            w2[w2_idx_not_to_match],
        ]
    )

    new_locations_w1 = get_new_idx_permutation(
        w1_idx_to_match, w1_idx_not_to_match, not_to_match_offset=0
    )

    new_locations_w2 = get_new_idx_permutation(
        w2_idx_to_match,
        w2_idx_not_to_match,
        not_to_match_offset=len(w1_idx_not_to_match),
    )

    return new_w, (new_locations_w1, new_locations_w2)
