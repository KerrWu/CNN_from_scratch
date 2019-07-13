def mat_mul(m1, m2):
    """
    matrix multiply implementation in python

    :param m1: matrix1, 2-d
    :param m2: matrix2, 2-d
    :return: result / None
    """

    if len(m1) == 0 or len(m2) == 0:
        return None

    if len(m1[0]) == 0 or len(m2[0]) == 0:
        return None

    m1_row = len(m1)
    m1_col = len(m1[0])

    m2_row = len(m2)
    m2_col = len(m2[0])

    assert m1_col == m2_row, "the last dimension of m1 and the first dimension of m2 must be equal"

    result = [[0] * m2_col for _ in range(m1_row)]

    for row in range(m1_row):

        for col in range(m2_col):
            cur_row = m1[row]
            cur_col = [elem[col] for elem in m2]

            cur_value = sum([x * y for x, y in zip(cur_row, cur_col)])
            result[row][col] = cur_value

    return result


m1 = [[1, 1, 1], [1, 1, 1]]
m2 = [[3, 2, 2], [3, 2, 2], [3, 2, 2]]

print(mat_mul(m1, m2))
