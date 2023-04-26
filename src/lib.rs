#![allow(non_snake_case)]

extern crate random_choice;
extern crate fnv;

use std::collections::{HashSet, HashMap};
use std::cmp::max;
use fnv::FnvHashMap;
use self::random_choice::random_choice;

pub struct GSDMM {
    alpha: f64,
    beta: f64,
    K:usize,
    V:f64,
    D:usize,
    maxit:isize,
    clusters: Vec<usize>,
    pub doc_vectors:Vec<Vec<usize>>,
    pub labels: Vec<usize>,
    pub cluster_counts: Vec<u32>,
    pub cluster_word_counts:Vec<u32>,
    pub word_index_map:HashMap<String, usize>,
    pub index_word_map:HashMap<usize, String>,
    pub cluster_word_distributions: Vec<FnvHashMap<usize,u32>>
}

impl GSDMM {
    pub fn new(alpha:f64, beta:f64, K: usize, maxit:isize, vocab:&HashSet<String>, docs:&Vec<Vec<String>>) -> GSDMM {
        let D = docs.len();

        // compute utilized vocabulary size.
        let mut word_index_map = HashMap::<String, usize>::with_capacity(vocab.len()/2);
        let mut index_word_map = HashMap::<usize, String>::with_capacity(vocab.len()/2);
        let mut index = 0_usize;
        let mut doc_vectors = Vec::<Vec<usize>>::with_capacity(D);
        for doc in docs {
            let mut doc_vector = Vec::<usize>::with_capacity(doc.len());
            for word in doc {
                if !word_index_map.contains_key(word) {
                    word_index_map.insert(word.clone(), index);
                    index_word_map.insert(index, word.clone());
                    index+=1;
                }
                doc_vector.push(*word_index_map.get(word).unwrap());
            }

            // dedupe vector and compact
            doc_vector.sort();
            doc_vector.dedup();
            doc_vector.shrink_to_fit();

            // stash
            doc_vectors.push(doc_vector);
        }
        let V = index as f64;
        println!("Fitting with alpha={}, beta={}, K={}, maxit={}, vocab size={}", alpha, beta, K, maxit, V as u32);

        let clusters = (0_usize..K).collect::<Vec<usize>>();
        let mut d_z: Vec<usize> = (0_usize..D).map(|_| 0_usize).collect::<Vec<usize>>(); // doc labels
        let mut m_z: Vec<u32> = GSDMM::zero_vector(K);  // cluster sizes
        let mut n_z: Vec<u32> = GSDMM::zero_vector(K);  // cluster word counts
        let mut n_z_w = Vec::<FnvHashMap<usize, u32>>::with_capacity(K);  // container for cluster word distributions
        for _ in 0_usize..K {
            let m = FnvHashMap::<usize, u32>::with_capacity_and_hasher(max(vocab.len() / 10, 100), Default::default());
            let _ = &n_z_w.push(m);
        }

        // randomly initialize cluster assignment
        let p = (0..K).map(|_| 1_f64 / (K as f64)).collect::<Vec<f64>>();

        let choices = random_choice().random_choice_f64(&clusters, &p, D) ;
        for i in 0..D {
            let z = *choices[i];
            let doc = &doc_vectors[i];
            d_z[i] = z;
            m_z[z] += 1;
            n_z[z] += doc.len() as u32;
            let clust_words: &mut FnvHashMap<usize, u32> = &mut n_z_w[z];
            for word in doc {
                if !clust_words.contains_key(word) {
                    clust_words.insert(*word, 0_u32);
                }
                    * clust_words.get_mut(word).unwrap() += 1_u32;
            }
        }

        GSDMM {
            alpha,
            beta,
            K,
            V,
            D,
            maxit,
            doc_vectors,
            clusters: clusters.clone(),     // Don't totally get why we need the clone here!
            labels: d_z,
            cluster_counts: m_z,
            cluster_word_counts: n_z,
            word_index_map,
            index_word_map,
            cluster_word_distributions: n_z_w
        }
    }

    pub fn fit(&mut self) {
        let mut number_clusters = self.K;
        for it in 0..self.maxit {
            let mut total_transfers = 0;
            for i in 0..self.D {
                let doc = &self.doc_vectors[i];
                let doc_size = doc.len() as u32;

                // remove the doc from its current cluster
                let z_old = self.labels[i];
                self.cluster_counts[z_old] -= 1;
                self.cluster_word_counts[z_old] -= doc_size;

                // modify the map: enclose it in a block so we can borrow views again.
                {
                    let old_clust_words: &mut FnvHashMap<usize, u32> = &mut self.cluster_word_distributions[z_old];
                    for word in doc {
                        *old_clust_words.get_mut(word).unwrap() -= 1_u32;

                        // compact dictionary once a key is exausted.
                        if old_clust_words[word] == 0_u32 {
                            old_clust_words.remove(word);
                        }
                    }
                }

                // update the probability vector
                let p = self.score(doc);

                // choose the next cluster randomly according to the computed probability
                let z_new: usize = *random_choice().random_choice_f64(&self.clusters, &p, 1)[0];

                // transfer document to the new cluster
                if z_new != z_old {
                    total_transfers += 1;
                }
                self.labels[i] = z_new;
                self.cluster_counts[z_new] += 1_u32;
                self.cluster_word_counts[z_new] += doc_size;

                {
                    let new_clust_words: &mut FnvHashMap<usize, u32> = &mut self.cluster_word_distributions[z_new];
                    for word in doc {
                        if !new_clust_words.contains_key(word) {
                            new_clust_words.insert(*word, 0_u32);
                        }
                            *new_clust_words.get_mut(word).unwrap() += 1_u32;
                    }
                }
            }
            let new_number_clusters = self.cluster_word_distributions.iter().map(|c| if !c.is_empty() {1} else {0} ).sum();
            println!("Iteration {}: {} docs transferred with {} clusters populated.", it, total_transfers, new_number_clusters);

            // apply ad-hoc convergence test
            if total_transfers==0 && new_number_clusters==number_clusters {
                println!("Converged after {} iterations. Solution has {} clusters.", it, new_number_clusters);
                break
            }
            number_clusters = new_number_clusters;
        }
    }

    pub fn score(&self, doc:&Vec<usize>) -> Vec<f64> {
        /// Score an input document using the formula of Yin and Wang 2014 (equation 3)
        /// http://dbgroup.cs.tsinghua.edu.cn/wangjy/papers/KDD14-GSDMM.pdf
        ///
        /// # Arguments
        ///
        /// * `doc` - A vector of unique index tokens characterizing the document
        ///
        /// # Value
        ///
        /// Vec<f64> - A length K probability vector where each component represents the probability
        /// of the doc belonging to a particular cluster.
        ///

        // We break the formula into the following pieces
        // p = N1*N2/(D1*D2) = exp(lN1 - lD1 + lN2 - lD2)
        // lN1 = log(m_z[z] + alpha)
        // lN2 = log(D - 1 + K*alpha)
        // lN2 = log(product(n_z_w[w] + beta)) = sum(log(n_z_w[w] + beta))
        // lD2 = log(product(n_z[d] + V*beta + i -1)) = sum(log(n_z[d] + V*beta + i -1))
        let mut p = (0..self.K).map(|_| 0_f64).collect::<Vec<f64>>();
        let lD1 = ((self.D - 1) as f64 + (self.K as f64) * self.alpha).ln();
        let doc_size = doc.len() as u32;
        for label in 0_usize..self.K {
            let lN1 = (self.cluster_counts[label] as f64 + self.alpha).ln();
            let mut lN2 = 0_f64;
            let mut lD2 = 0_f64;

            let cluster: &FnvHashMap<usize, u32> = &self.cluster_word_distributions[label];

            for word in doc {
                lN2 += (*cluster.get(word).unwrap_or(&0_u32) as f64 + self.beta).ln();
            }
            for j in 1_u32..(doc_size+1) {
                lD2 += ((self.cluster_word_counts[label] + j) as f64 - 1_f64 + self.V * self.beta).ln();
            }
            p[label] = (lN1 - lD1 + lN2 - lD2).exp();
        }

        // normalize the probability
        let pnorm: f64 = p.iter().sum();
        if pnorm>0_f64 {
            for label in 0_usize..self.K {
                p[label] /= pnorm;
            }
        }
        p
    }

    fn zero_vector(size:usize) -> Vec<u32>
    {
        let mut v = Vec::<u32>::with_capacity(size);
        for _ in 0_usize..size {
            v.push(0_u32)
        }
        v
    }

}

#[test]
fn simple_run() {
    let mut vocab = HashSet::<String>::new();
    vocab.insert("A".to_string());
    vocab.insert("B".to_string());
    vocab.insert("C".to_string());

    let mut docs = Vec::<Vec<String>>::new();
    docs.push(vec!("A".to_string()));
    docs.push(vec!("A".to_string()));
    docs.push(vec!("B".to_string()));
    docs.push(vec!("B".to_string()));
    docs.push(vec!("B".to_string()));
    docs.push(vec!("B".to_string()));
    docs.push(vec!("B".to_string()));
    docs.push(vec!("B".to_string()));
    docs.push(vec!("B".to_string()));
    docs.push(vec!("B".to_string()));
    docs.push(vec!("C".to_string()));
    docs.push(vec!("C".to_string()));
    docs.push(vec!("C".to_string()));
    docs.push(vec!("C".to_string()));
    docs.push(vec!("C".to_string()));
    docs.push(vec!("C".to_string()));
    docs.push(vec!("C".to_string()));
    docs.push(vec!("C".to_string()));

    let mut model = GSDMM::new(0.1, 0.00001, 10, 30, vocab, docs);
    model.fit();

    // check the total number across all partitions is equal to the number of docs
    assert_eq!(18, model.cluster_counts.iter().sum::<u32>());

    // check that we get three clusters
    assert_eq!(3, model.cluster_counts.into_iter().filter(|x| x>&0_u32 ).collect::<Vec<u32>>().len());

    // check that the clusters are pure
    let mut check_map = HashMap::<usize,String>::new();
    for (i,label) in vec!("A","A","B","B","B","B","B","B","B","B","C","C","C","C","C","C","C","C").into_iter().enumerate() {
        if check_map.contains_key(&model.labels[i]) {
            assert_eq!(check_map[&model.labels[i]], label);
        } else {
            check_map.insert(model.labels[i], label.to_string());
        }

    }
}

#[test]
fn indexing() {
    let mut vocab = HashSet::<String>::new();
    vocab.insert("A".to_string());
    vocab.insert("B".to_string());
    vocab.insert("C".to_string());
    vocab.insert("D".to_string());

    let mut docs = Vec::<Vec<String>>::new();
    docs.push(vec!("A".to_string(),"B".to_string()));
    docs.push(vec!("D".to_string()));
    docs.push(vec!("C".to_string()));

    let mut model = GSDMM::new(0.1, 0.00001, 10, 30, vocab, docs);

    // test the index mapping
    assert_eq!("A", model.index_word_map.get(&0_usize).unwrap());
    assert_eq!("B", model.index_word_map.get(&1_usize).unwrap());
    assert_eq!("C", model.index_word_map.get(&3_usize).unwrap());
    assert_eq!("D", model.index_word_map.get(&2_usize).unwrap());
    assert_eq!(0_usize, *model.word_index_map.get("A").unwrap());
    assert_eq!(1_usize, *model.word_index_map.get("B").unwrap());
    assert_eq!(2_usize, *model.word_index_map.get("D").unwrap());
    assert_eq!(3_usize, *model.word_index_map.get("C").unwrap());

}
