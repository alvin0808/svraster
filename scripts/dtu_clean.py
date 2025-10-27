import logging, traceback, trimesh
trimesh.util.attach_to_log(level=logging.ERROR)  # �ʿ� �� DEBUG��

# ������ �籸��: process=False�� ����� ���� Ŭ����
mesh = trimesh.Trimesh(new_vertices, new_faces, process=False)

# 1) ������ Ŭ����
mesh.remove_duplicate_faces()
mesh.remove_degenerate_faces()
mesh.remove_unreferenced_vertices()
trimesh.repair.fix_inversion(mesh)  # ������ face ����ȭ
trimesh.repair.fix_winding(mesh)    # winding ����
trimesh.repair.fix_normals(mesh)

# 2) NaN/Inf ���� (������ ������ ��ź)
if np.isnan(mesh.vertices).any() or np.isinf(mesh.vertices).any():
    bad = ~np.any(~np.isfinite(mesh.vertices), axis=1)
    mesh.update_vertices(bad)  # �ʿ�� ���� ����ŷ/���ε���

# 3) split�� �����ϸ� �������� largest CC ���
try:
    print('Kept only the largest CC (split)')
    parts = mesh.split(only_watertight=False)
    if len(parts) == 0:
        raise RuntimeError("split returned 0 parts")
    mesh = max(parts, key=lambda m: len(m.faces))
except Exception:
    print('split() failed; falling back to adjacency-based selection')
    traceback.print_exc()

    # face adjacency�� ������ ���� ���ϱ�
    adj = mesh.face_adjacency
    if adj is None or len(adj) == 0:
        # ������ ���� ������, ���� ���� face�� ���� ���ڱ� �ڽš��� �� �ִ� ������
        # Ȥ�� �� face���̶�� ��ü�� �����ϰų�, �ּ� face �� �̸��� �����ϴ� ��å ��1
        pass
    else:
        comps = trimesh.graph.connected_components(adj,
                                                   nodes=np.arange(len(mesh.faces)),
                                                   min_len=1)
        # comps�� set���� list
        largest = max(comps, key=len)
        faces_keep = np.array(sorted(list(largest)), dtype=np.int64)
        # submesh�� ����
        mesh = mesh.submesh([faces_keep], only_watertight=False, append=False)[0]

# ������: Ȥ�� ���� ��� ����
mesh.remove_unreferenced_vertices()
