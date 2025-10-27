import logging, traceback, trimesh
trimesh.util.attach_to_log(level=logging.ERROR)  # 필요 시 DEBUG로

# 안전한 재구성: process=False로 만들고 수동 클린업
mesh = trimesh.Trimesh(new_vertices, new_faces, process=False)

# 1) 가벼운 클린업
mesh.remove_duplicate_faces()
mesh.remove_degenerate_faces()
mesh.remove_unreferenced_vertices()
trimesh.repair.fix_inversion(mesh)  # 뒤집힌 face 정상화
trimesh.repair.fix_winding(mesh)    # winding 정리
trimesh.repair.fix_normals(mesh)

# 2) NaN/Inf 정리 (있으면 무조건 폭탄)
if np.isnan(mesh.vertices).any() or np.isinf(mesh.vertices).any():
    bad = ~np.any(~np.isfinite(mesh.vertices), axis=1)
    mesh.update_vertices(bad)  # 필요시 직접 마스킹/재인덱싱

# 3) split이 실패하면 수동으로 largest CC 계산
try:
    print('Kept only the largest CC (split)')
    parts = mesh.split(only_watertight=False)
    if len(parts) == 0:
        raise RuntimeError("split returned 0 parts")
    mesh = max(parts, key=lambda m: len(m.faces))
except Exception:
    print('split() failed; falling back to adjacency-based selection')
    traceback.print_exc()

    # face adjacency로 연결요소 직접 구하기
    adj = mesh.face_adjacency
    if adj is None or len(adj) == 0:
        # 인접이 전혀 없으면, 가장 많은 face를 가진 “자기 자신”이 곧 최대 연결요소
        # 혹은 고립 face들이라면 전체를 유지하거나, 최소 face 수 미만은 제거하는 정책 택1
        pass
    else:
        comps = trimesh.graph.connected_components(adj,
                                                   nodes=np.arange(len(mesh.faces)),
                                                   min_len=1)
        # comps는 set들의 list
        largest = max(comps, key=len)
        faces_keep = np.array(sorted(list(largest)), dtype=np.int64)
        # submesh로 추출
        mesh = mesh.submesh([faces_keep], only_watertight=False, append=False)[0]

# 마무리: 혹시 남은 찌꺼기 정리
mesh.remove_unreferenced_vertices()
