import math

import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
import datetime as dt

class kmknn_v3:
    N = 0
    s = 0
    M = 0

    # clusters
    Cj = None
    # cluster centers
    cj = []
    # radius cluster
    rj = None
    rj_sorted_index = None
    # DISTANCES yji_cj*
    d_yji_cj = []

    cluster_labels = None
    cluster_labels = None
    qtd_euclidian_calc = 0

    Tempo = None

    def offline(self):

        T = [
                [0.5, 0.8, 0.8, 0.9, 0.8, 0.7, 0.9, 0.5, 0.7, 0.4, 0.8, 0.8, 0.6, 0.5, 0.6, 0.9, 0.8, 0.8, 1.0, 1.0, 0.8, 0.6, 1.0], # Ciência da Computação
                [0.8, 1.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.7, 0.7, 0.7, 0.8, 0.8, 0.8, 0.7, 0.8, 1.0, 1.0, 0.8, 1.0, 1.0, 0.6, 0.6, 1.0], # Eng. da Computação
                [0.7, 0.9, 0.8, 0.7, 0.7, 0.6, 0.6, 0.6, 0.7, 0.6, 0.7, 0.7, 0.7, 0.6, 0.6, 0.3, 0.4, 0.8, 0.8, 1.0, 1.0, 0.4, 1.0], # ADS
                [0.8, 0.7, 0.8, 0.9, 0.9, 0.9, 0.7, 0.8, 0.9, 0.7, 0.9, 0.7, 0.8, 0.7, 0.7, 0.3, 0.4, 1.0, 0.8, 1.0, 0.8, 0.6, 0.8], # Gestão de T.I   
                [0.9, 0.9, 1.0, 0.5, 0.9, 1.0, 0.8, 0.8, 0.6, 0.7, 0.8, 0.9, 0.7, 0.8, 0.7, 0.3, 0.4, 1.0, 1.0, 1.0, 0.6, 1.0, 0.8], # Seg. Informação                  
                [0.6, 0.9, 0.8, 0.8, 0.8, 0.8, 0.8, 0.6, 0.7, 0.7, 0.7, 0.8, 0.8, 0.6, 0.7, 0.2, 0.3, 0.6, 1.0, 1.0, 0.4, 1.0, 0.8]  # IA
            ]

        self.N = len(T)
        self.s = 1  # determinação do autor como melhor valor indicado
        self.M = (int)(self.s * math.sqrt(self.N))
        self.Cj = np.arange(self.M)
        # radius cluster
        self.rj = np.empty(shape=self.M)
        self.rj_sorted_index = np.empty(shape=self.M)
        # DISTANCES yji_cj*
        self.d_yji_cj = np.empty(shape=self.N)


        """ ESTÁGIO DE CONSTRUÇÃO """
        kmeans = KMeans(n_clusters=self.M, init='k-means++').fit(T)
        self.cluster_labels = kmeans.labels_
        self.cluster_centers = kmeans.cluster_centers_

        self.cj = self.cluster_centers

        # Rj calculate
        for i in range(self.M):
            yij_rj = np.where(self.cluster_labels == i)
            distance_yij_rj = 0
            c_rj = self.cj[i]
            for j in yij_rj[0]:
                compute_rj = self.euclidian_distance(c_rj, T[j])
                if compute_rj > distance_yij_rj:
                    distance_yij_rj = compute_rj
            self.rj[i] = distance_yij_rj
        rj_sorted_index = np.argsort(self.rj)

        # d_yji_cj calculate
        for i in range(self.M):
            yij_cj = np.where(self.cluster_labels == i)
            _cj = self.cj[i]
            for j in yij_cj[0]:
                compute_yji_cj = self.euclidian_distance(T[j], _cj)
                self.d_yji_cj[j] = compute_yji_cj


    def online(self, q, K, verbose):

        T = [
                [0.5, 0.8, 0.8, 0.9, 0.8, 0.7, 0.9, 0.5, 0.7, 0.4, 0.8, 0.8, 0.6, 0.5, 0.6, 0.9, 0.8, 0.8, 1.0, 1.0, 0.8, 0.6, 1.0], # Ciência da Computação
                [0.8, 1.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.7, 0.7, 0.7, 0.8, 0.8, 0.8, 0.7, 0.8, 1.0, 1.0, 0.8, 1.0, 1.0, 0.6, 0.6, 1.0], # Eng. da Computação
                [0.7, 0.9, 0.8, 0.7, 0.7, 0.6, 0.6, 0.6, 0.7, 0.6, 0.7, 0.7, 0.7, 0.6, 0.6, 0.3, 0.4, 0.8, 0.8, 1.0, 1.0, 0.4, 1.0], # ADS
                [0.8, 0.7, 0.8, 0.9, 0.9, 0.9, 0.7, 0.8, 0.9, 0.7, 0.9, 0.7, 0.8, 0.7, 0.7, 0.3, 0.4, 1.0, 0.8, 1.0, 0.8, 0.6, 0.8], # Gestão de T.I   
                [0.9, 0.9, 1.0, 0.5, 0.9, 1.0, 0.8, 0.8, 0.6, 0.7, 0.8, 0.9, 0.7, 0.8, 0.7, 0.3, 0.4, 1.0, 1.0, 1.0, 0.6, 1.0, 0.8], # Seg. Informação                  
                [0.6, 0.9, 0.8, 0.8, 0.8, 0.8, 0.8, 0.6, 0.7, 0.7, 0.7, 0.8, 0.8, 0.6, 0.7, 0.2, 0.3, 0.6, 1.0, 1.0, 0.4, 1.0, 0.8]  # IA
            ]

        
        self.qtd_euclidian_calc = 0
        """ESTÁGIO DE PESQUISA"""
        """OUTPUT"""
        y_sNN = np.array(self.N)
        d_x_sNN = np.array(self.N)
        """variaveis"""
        n = 0
        dk = 0
        # compute euclidian
        d_x_cj = np.zeros(shape=self.M)
        for j in range(self.M):
            compute_x_cj = self.euclidian_distance(q, self.cj[j])
            d_x_cj[j] = compute_x_cj
            self.qtd_euclidian_calc += 1

        d_x_cj_sorted_ind = np.argsort(d_x_cj)

        # compute euclidian
        y1i_c1 = np.where(self.cluster_labels == d_x_cj_sorted_ind[0])
        d_y1i_c1 = []
        for i in y1i_c1[0]:
            compute_x_y1i = self.euclidian_distance(q, T[i])
            d_y1i_c1.append(compute_x_y1i)
            self.qtd_euclidian_calc += 1

        d_y1i_c1_sorted_ind = np.argsort(d_y1i_c1)
        d_y1i_c1_np = np.array(d_y1i_c1)

        N1 = len(y1i_c1[0])
        if K <= N1:
            d_x_sNN = np.append(d_x_sNN, d_y1i_c1_np[d_y1i_c1_sorted_ind[:K]])
            y_sNN = np.append(y_sNN, y1i_c1[0][d_y1i_c1_sorted_ind[:K]])
        else:
            d_x_sNN = np.append(d_x_sNN, d_y1i_c1_np[d_y1i_c1_sorted_ind[:N1]])
            y_sNN = np.append(y_sNN, y1i_c1[0][d_y1i_c1_sorted_ind[:N1]])

        n = K
        dk = d_x_sNN[-1]

        for j in range(self.M):
            if j != d_x_cj_sorted_ind[0]:
                if d_x_cj[j] - self.rj[j] < dk:
                    Nj = np.where(self.cluster_labels == j)
                    for i in Nj[0]:
                        d_yji = self.d_yji_cj[i]
                        yji = T[i]
                        if np.abs(d_x_cj[j] - d_yji) <= dk:
                            # compute distances
                            d_x_yji = self.euclidian_distance(q, yji)
                            self.qtd_euclidian_calc += 1
                            if (d_x_yji < dk) or np.abs(d_x_yji - dk) <= 0.0001:
                                y_sNN = np.append(y_sNN, i)
                                d_x_sNN = np.append(d_x_sNN, d_x_yji)
                                n += 1

                d_x_sNN_sorted_ind = np.argsort(d_x_sNN)

                y_sNN = y_sNN[d_x_sNN_sorted_ind[:K]]
                d_x_sNN = d_x_sNN[d_x_sNN_sorted_ind[:K]]
                n = K
                dk = d_x_sNN[-1]

        if verbose:
            print(" --- VIZINHOS MAIS PRÓXIMOS --- ")
            print(y_sNN)
            print(" --- DISTANCIAS DOS VIZINHOS MAIS PRÓXIMOS --- ")
            print(d_x_sNN)
            print(" --- QTD CALCULOS DE DISTANCIAS --- ")
            print(self.qtd_euclidian_calc)
        return self.FinalRecomendation(d_x_sNN, y_sNN,)
        #return [d_x_sNN, y_sNN, self.qtd_euclidian_calc]

    # CALCULA A DISTANCIA EUCLIDIANA PARA CADA UM DOS PONTOS A PARTIR DA ENTRADA (input)
    def euclidian_distance(self, v1, v2):
        distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(v1, v2)]))
        return distance


    def crossValidation(self, k, s, dataset):
        ds = dataset
        kf = KFold(n_splits=s)
        ds = np.array(ds)
        qtd_calc_euclidian = 0

        m = 0
        Q = len(ds)
        N = 0
        q = 0  # Contador de buscas. Usado para expressar o percentual de trabalho realizado.

        tInicio = dt.datetime.now()  # Tempo de início do experimento.
        for train, test in kf.split(ds):
            ### RUN OFFLINE
            self.offline(ds[train])
            ###
            N = len(train)

            for x in ds[test]:
                d, knn, qtd_calc_euc = self.online(ds[train], x, k, False)
                qtd_calc_euclidian += qtd_calc_euc
                q += 1


            # soma dos tempos de busca
            self.Tempo = dt.datetime.now() - tInicio
            print("{}: {:.0f}%".format(self.Tempo, 100 * q / Q), flush=True)
            print("RRDC")
            # RRDC = (1 - qtd_calc_euclidian / (Q * N)) * 100
            # print(Q * N)
            RRDC = (1 - qtd_calc_euclidian / (Q * N)) * 100
            print(RRDC)
        print("GERAL EUCLIDIAN")
        print(qtd_calc_euclidian)

    def FinalRecomendation(self, d, indices):
        c = [
                "Ciência da Computação", "Engenharia da Computação", "ADS",
                "Gestão da TI", "Segurança da Informação", "Inteligência Artificial"
            ]
        
        recomendacoes = []
        for i in range(len(indices)):
            try:
                idx_curso = int(indices[i])
                if idx_curso < len(c):
                    recomendacoes.append({
                        "Curso": c[idx_curso],
                        "Distancia": round(float(d[i]), 4)
                    })
            except (IndexError, ValueError):
                continue
        return recomendacoes

if __name__ == "__main__":
    # 1. Simulação do Dataset de Cursos (T) - Conforme Definição 2.1 [3]
    # N=6 instâncias, D=15 atributos (Valores reais entre 0.0 e 1.0)
    T = [
        [0.9, 0.4, 0.9, 0.2, 0.4, 0.6, 0.3, 0.5, 0.4, 0.7, 0.5, 0.2, 1.0, 0.4, 0.4], # Ciência da Computação
        [0.9, 1.0, 0.7, 0.3, 0.3, 0.4, 0.5, 1.0, 0.3, 0.4, 1.0, 0.2, 0.6, 0.2, 0.3], # Eng. da Computação
        [0.4, 0.1, 1.0, 0.4, 0.4, 0.3, 0.5, 0.2, 0.9, 0.3, 0.1, 0.4, 0.3, 0.9, 0.8], # ADS
        [0.2, 0.1, 0.3, 1.0, 0.4, 0.3, 0.5, 0.1, 1.0, 0.2, 0.1, 1.0, 0.2, 0.5, 0.7], # Gestão da TI
        [0.4, 0.3, 0.6, 0.3, 1.0, 0.4, 0.9, 0.3, 0.4, 0.3, 0.2, 0.7, 0.3, 0.3, 0.9], # Seg. Informação
        [0.9, 0.3, 0.8, 0.2, 0.3, 1.0, 0.3, 0.4, 0.4, 1.0, 0.4, 0.2, 0.8, 0.3, 0.5]  # IA
    ]

    cursos_labels = [
        "Ciência da Computação", "Engenharia da Computação", "ADS",
        "Gestão da TI", "Segurança da Informação", "Inteligência Artificial"
    ]

    # 2. Instância de busca x (Perfil de interesse do Aluno)
    # Exemplo: Aluno com interesse em Hardware (Atrib 2) e Física (Atrib 11)
    #x = [0.9, 0.4, 0.9, 0.2, 0.9, 0.6, 0.3, 0.5, 0.4, 0.7, 0.5, 0.2, 1.0, 0.4, 0.4]

    # 3. Inicialização do Algoritmo conforme sua classe [4]
    # k=1 para encontrar o curso mais similar (ENN - Exact Nearest Neighbor) [5, 6]
    modelo = kmknn_v3()

    # 4. FASE OFFLINE: Estágio de Construção [1, 7]
    # Induz o agrupamento (M = sqrt(N)) usando k-Means++ [8, 9]
    print("Executando Fase Offline (Agrupamento)...")
    modelo.offline(T)

    # 5. FASE ONLINE: Estágio de Pesquisa [2, 10]
    # Parâmetros: dataset T, busca x, vizinhos K=1, verbose=True
    print("Executando Fase Online (Busca Acelerada)...")
    resultado = modelo.online(T, x, 1, verbose=True)

    # 6. Processamento do Output [11]
    # O método online retorna uma lista: [distâncias, índices, qtd_cálculos]
    dists, indices, qtd_calc = resultado

    print("\n--- RECOMENDAÇÃO FINAL ---")
    # Como y_sNN no seu código armazena índices [12], mapeamos para o nome do curso
    for i in range(len(indices)):
        # Nota: O seu código usa np.append em arrays pré-alocados,
        # garantimos aqui a limpeza do índice para exibição
        try:
            idx_curso = int(indices[i])
            if idx_curso < len(cursos_labels):
                print(f"Curso: {cursos_labels[idx_curso]}")
                print(f"Distância: {dists[i]:.4f}")
        except:
            pass