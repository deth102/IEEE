# export_rf_to_c.py
from Model import RandomForest
from prepare_data import X_train, y_train


# ======================================================
# Export ONE decision tree to C (compact format)
# ======================================================
def export_tree_to_c(tree, tree_id):
    nodes = []
    node_index = {}

    def traverse(node):
        node_index[id(node)] = len(nodes)
        nodes.append(node)
        if node.value is None:
            traverse(node.left)
            traverse(node.right)

    traverse(tree.root)

    lines = []
    lines.append(f"/* ---- Tree {tree_id} ---- */")
    lines.append(f"static const TreeNode tree{tree_id}_nodes[] = {{")

    row = []
    for node in nodes:
        if node.value is None:
            l = node_index[id(node.left)]
            r = node_index[id(node.right)]
            row.append(f"{{{node.feature},{node.threshold:.6f}f,{l},{r},-1}}")
        else:
            row.append(f"{{-1,0,-1,-1,{node.value}}}")

    lines.append("    " + ",".join(row))
    lines.append("};\n")

    return "\n".join(lines), len(nodes)


# ======================================================
# Export Random Forest to C
# ======================================================
def export_forest_to_c(rf, num_classes, var_name="rf_model"):
    output = []
    tree_sizes = []

    for i, tree in enumerate(rf.trees):
        code, n = export_tree_to_c(tree, i)
        output.append(code)
        tree_sizes.append(n)

    output.append("/* ---- Forest ---- */")
    output.append("static const DecisionTree forest_trees[] = {")
    output.append(
        "    " + ",".join(
            f"{{tree{i}_nodes,{n}}}" for i, n in enumerate(tree_sizes)
        )
    )
    output.append("};\n")

    output.append(f"static const RandomForest {var_name} = {{")
    output.append(f"    forest_trees,{len(rf.trees)},{num_classes}")
    output.append("};\n")

    return "\n".join(output)


# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":
    rf = RandomForest(n_trees=10, max_depth=7, max_leaf_nodes=22, random_state=42)
    rf.fit(X_train, y_train)

    NUM_CLASSES = 5

    c_code = export_forest_to_c(rf, NUM_CLASSES)

    with open("rf_model.c.txt", "w") as f:
        f.write(c_code)

    print("Exported compact Random Forest to rf_model.c.txt")
