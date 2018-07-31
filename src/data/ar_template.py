from data.path_generator import TemplateData


class TemplateMesh(object):
    def __init__(self, _template_img="qrmarker.png", w_split=8, h_split=8, _nneighbor=4, image=None):
        self.h, self.w = image.shape[:2]
        self.w_split = w_split
        self.h_split = h_split
        self.offset_c = self.w // self.w_split
        self.offset_r = self.h // self.h_split
        self.nneighbor = _nneighbor
        self.data_path = TemplateData(_template_img)

    def get_faces_num(self):
        return self.w_split * self.h_split

    def make_polygon_color_map(self):
        import cv2
        import numpy as np
        img = np.zeros((self.h, self.w, 1), np.uint8)
        for i in range(self.get_faces_num()):
            y, x = self.calculate_mesh_topleftvertex(i)
            mesh = np.tile(np.uint8([i]), (self.offset_r, self.offset_c, 1))
            img[y:y+self.offset_r, x:x+self.offset_c] = mesh

        return img

    def calculate_mesh_topleftvertex(self, id):
        ## coluculate mash top left vertex pt = [Y-axis, X-axis]
        div_r = id // self.w_split
        mod_c = id % self.w_split

        return self.offset_r * div_r, self.offset_c * mod_c

    def calculate_face_corners_index(self, id):
        return id + 0, id + 1, id + self.w, id + self.w_split + 1

    def calculate_mesh_corners(self, id):
        #メッシュ番号を入力メッシュを構成する頂点を返す
        import numpy as np
        y, x = self.calculate_mesh_topleftvertex(id)

        def correct_length(val, threshold):
            if val > threshold:
                return threshold - 1
            elif val < 0:
                return 0
            else:
                return val

        return np.float32([[x, y], [correct_length(x + self.offset_c, self.w), y],
                           [correct_length(x + self.offset_c, self.w), correct_length(y + self.offset_r, self.h)],
                           [x, correct_length(y + self.offset_r, self.h)]])

    def get_meshid_index(self, id):
        #行列のインデックスを返す
        div_r = id // self.w_split
        mod_c = id % self.w_split
        return div_r, mod_c

    def get_mesh_map(self):
        import numpy as np
        mesh_ids = np.arange(self.get_faces_num()).reshape(self.h_split, self.w_split)
        return mesh_ids

    def get_mesh_shape(self):
        return self.h_split, self.w_split

    def get_meshidlist_nneighbor(self, id):
        """
        直線リストにおける メッシュIDの近傍に属するIDを取得する
        :param id:
        :return:
        """

        def validate(x):
            if x < 0:
                return None
            else:
                return x
        def v_(y, *args):
            a = list(args)
            if int(id/self.w_split) == 0:
                #上辺
                a[1] = -1
            if int(id%self.w_split) == 0:
                #左辺
                a[0] = -1
            if int(id/self.w_split) == self.h_split -1:
                #下辺
                a[3] = -1
            if int(id%self.w_split) == self.w_split -1:
                #サ変
                a[2] = -1
            return a

        if self.nneighbor == 4:
            a = [id - 1, id - self.w_split, id + 1, id + self.w_split]
            return list(map(validate, v_(id, *a)))
        elif self.nneighbor == 8:
            a = [id - 1, id - 1 - self.w_split,
                 id - self.w_split, id + 1 - self.w_split, id + 1,
                 id + 1 + self.w_split, id + self.w_split, id + self.w_split - 1]
            return list(map(validate, v_(a)))
        elif self.nneighbor == 3:
            """三角メッシュ"""
            if id % 2 == 0:
                """上三角"""
                a = [id - 1, id - (self.w_split - 1), id + 1]
                return list(map(validate, v_(a)))
            else:
                """下三角"""
                a = [id - 1, id + 1, id + (self.w_split - 1)]
                return list(map(validate, v_(a)))

    def get_meshidlist_8neighbor(self, id):
        a = [id - 1, id - 1 - self.w_split,
             id - self.w_split, id + 1 - self.w_split, id + 1,
             id + 1 + self.w_split, id + self.w_split, id + self.w_split - 1]

        def validate(x):
            if x < 0:
                return None
            else:
                return x

        def v_(*args):
            a = list(args)
            if int(id/self.w_split) == 0:
                #上辺
                a[1] = -1
                a[2] = -1
                a[3] = -1
            if int(id%self.w_split) == 0:
                #左辺
                a[1] = -1
                a[0] = -1
                a[7] = -1
            if int(id/self.w_split) == self.w_split -1:
                #下辺
                a[5] = -1
                a[6] = -1
                a[7] = -1
            if int(id%self.w_split) == self.w_split -1:
                #右辺
                a[3] = -1
                a[4] = -1
                a[5] = -1
            return a
        return list(map(validate, v_(*a)))

    def get_mesh_recanglarvertex_list(self, list_id):
        list_mesh_vertex = []
        for id in list_id:
            list_mesh_vertex.append(self.calculate_mesh_corners(id))
        return list_mesh_vertex

    def get_nodes(self):
        return list(int(i) for i in range((self.srows + 1)*(self.scols + 1)))

    def get_meshid_list_with_node(self, node_id):
        node_cols = self.w_split + 1
        x = int(node_id/node_cols)
        has_meshes = [
            node_id - x,
            node_id - x -1,
            node_id - x - 9,
            node_id - x - 8
        ]

        def validate(x):
            if x < 0:
                return None
            else:
                return int(x)

        def v_(*args):
            a = list(args)
            if int(node_id/node_cols) == 0:
                #上辺
                a[2] = -1
                a[3] = -1
            if int(node_id%node_cols) == 0:
                #左辺
                a[2] = -1
                a[1] = -1
            if int(node_id/node_cols) == self.h_split:
                #下辺
                a[0] = -1
                a[1] = -1
            if int(node_id%node_cols) == self.w_split:
                #右辺
                a[0] = -1
                a[3] = -1
            return a

        return list(map(validate, v_(*has_meshes)))

    def get_nodes_has_meshes_id(self):
        nodes = self.get_nodes()
        return list(self.get_meshid_list_with_node(node) for node in nodes)

if __name__ == "__main__":
    h = TemplateMesh.get_nodes()