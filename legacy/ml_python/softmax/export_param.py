# import pickle
# import numpy as np

# with open("softmax_logistic_model.pkl", "rb") as f:
#     model = pickle.load(f)

# W = model.W        # (C, D)
# b = model.b        # (C,)
# mean = model.mean  # (D,)
# std = model.std    # (D,)

# np.save("W.npy", W)
# np.save("b.npy", b)
# np.save("mean.npy", mean)
# np.save("std.npy", std)
# def export_array(name, arr):
#     flat = arr.flatten()
#     print(f"static const float {name}[{len(flat)}] = {{")
#     for v in flat:
#         print(f"  {v:.8e},")
#     print("};\n")

# C, D = model.W.shape

# print(f"#define NUM_CLASSES {C}")
# print(f"#define NUM_FEATURES {D}\n")

# export_array("W", model.W)
# export_array("b", model.b)
# export_array("feature_mean", model.mean)
# export_array("feature_std", model.std)
import pickle
import numpy as np

with open("softmax_logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

W = model.W        # (C, D)
b = model.b        # (C,)
mean = model.mean  # (D,)
std = model.std    # (D,)

C, D = W.shape

print(f"#define NUM_CLASSES {C}")
print(f"#define NUM_FEATURES {D}\n")


def export_matrix(name, mat):
    print(f"const float {name}[NUM_CLASSES][NUM_FEATURES] = {{")
    for row in mat:
        row_str = ", ".join(f"{v:.8e}" for v in row)
        print(f"  {{ {row_str} }},")
    print("};\n")


def export_vector(name, vec):
    vec_str = ", ".join(f"{v:.8e}" for v in vec)
    print(f"const float {name}[{len(vec)}] = {{")
    print(f"  {vec_str}")
    print("};\n")


export_matrix("W", W)
export_vector("b", b)
export_vector("mean", mean)
export_vector("feature_std", std)
