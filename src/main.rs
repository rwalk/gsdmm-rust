extern crate docopt;
extern crate rustc_serialize;
extern crate random_choice;

use docopt::Docopt;
use std::io::{BufRead,BufReader};
use std::fs::File;
use std::collections::{HashSet, HashMap};
use std::cmp::max;
use std::io::Write;
use self::random_choice::random_choice;

const USAGE: &'static str ="
Gibbs sampling algorithm for a Dirichlet Mixture Model of Yin and Wang 2014.

Usage:
  gsdmm <datafile> <vocabfile> <labelout> <clusterout> [-k <max_clusters>] [-a <alpha>] [-b <beta>] [-m <maxit>]
  gsdmm (-h | --help)
  gsdmm --version

Options:
  -h --help             Show this screen.
  --version             Show version.
  -k=<K>                Upper bound on the number of possible clusters. [default: 8]
  -a --alpha=<alpha>    Alpha controls the probability that a student will join a table that is currently empty
                        When alpha is 0, no one will join an empty table. [default: 0.1]
  -b --beta=<beta>      Beta controls the student's affinity for other students with similar interests. A low beta means
                        that students desire to sit with students of similar interests. A high beta means they are less
                        concerned with affinity and are more influenced by the popularity of a table. [default: 0.1]
  -m --maxit=<m>        Maximum number of iterations. [default: 30]

";

#[derive(Debug, RustcDecodable)]
struct Args {
    //    flag_mode: isize,
    arg_datafile: String,
    arg_vocabfile: String,
    arg_labelout: String,
    arg_clusterout: String,
    flag_k: isize,
    flag_alpha: f64,
    flag_beta: f64,
    flag_maxit: isize
}

fn main() {

    let args: Args = Docopt::new(USAGE)
        .and_then(|d| d.decode())
        .unwrap_or_else(|e| e.exit());

    // get the data and vocabulary
    let vocab:HashSet<String> = lines_from_file(&args.arg_vocabfile).into_iter().collect();
    let docs:Vec<Vec<String>> = lines_from_file(&args.arg_datafile).into_iter().map(|line| {
        line.to_owned().split_whitespace().map(|s| s.to_owned()).filter(|s| (&vocab).contains(s)).collect::<Vec<String>>()
    }).collect::<Vec<Vec<String>>>();

    // setup algorithm containers
    let alpha = args.flag_alpha;
    let beta = args.flag_beta;
    let K = args.flag_k as usize;
    let V = vocab.len() as f64;
    let D = docs.len();
    println!("Fitting with alpha={}, beta={}, K={}, maxit={}.", alpha, beta, K, args.flag_maxit);

    let labels = (0_usize..K).collect::<Vec<usize>>();
    let mut d_z:Vec<usize>  = (0_usize..D).map(|_| 0_usize).collect::<Vec<usize>>(); // doc labels
    let mut m_z:Vec<u32>  = zero_vector(K);  // cluster sizes
    let mut n_z:Vec<u32>  = zero_vector(K);  // cluster word counts
    let mut n_z_w = Vec::<HashMap<String,u32>>::with_capacity(K);  // container for cluster word distributions
    for _ in 0_usize..K {
        let mut m = HashMap::<String,u32>::with_capacity(max(vocab.len()/10, 100));
        &n_z_w.push(m);
    }

    // randomly initialize cluster assignment
    let mut p = (0..K).map(|_| 1_f64/(K as f64)).collect::<Vec<f64>>();
    let choices = random_choice().random_choice_f64(&labels, &p, D);
    for i in 0..D {
        let z = choices[i].clone();
        let ref doc = docs[i];
        d_z[i] = z;
        m_z[z] += 1;
        n_z[z] += doc.len() as u32;
        let ref mut clust_words:HashMap<String,u32> = n_z_w[z];
        for word in doc {
            if !clust_words.contains_key(word) {
                clust_words.insert(word.clone(), 0_u32);
            }
                *clust_words.get_mut(word).unwrap() += 1_u32;
        }
    }

    let mut number_clusters = K;
    for it in 0..args.flag_maxit {
        let mut total_transfers = 0;
        for i in 0..D {
            let ref doc = docs[i];
            let doc_size = doc.len() as u32;

            // remove the doc from its current cluster
            let z_old = d_z[i];
            m_z[z_old] -= 1;
            n_z[z_old] -= doc_size;

            // modify the map: enclose it in a block so we can borrow views again.
            {
                let ref mut old_clust_words: HashMap<String, u32> = n_z_w[z_old];
                for word in doc {
                    *old_clust_words.get_mut(word).unwrap() -= 1_u32;

                    // compact dictionary once a key is exausted.
                    if old_clust_words[word] == 0_u32 {
                        old_clust_words.remove(word);
                    }
                }
            }

            // update the probability vector
            let lD1 = ((D - 1) as f64 + (K as f64) * alpha).ln();
            for label in 0_usize..K {
                let N1 = m_z[label] as f64 + alpha;
                let lN1 = if N1 > 0_f64 { N1.ln() } else { 0_f64 };
                let mut lN2 = 0_f64;
                let mut lD2 = 0_f64;

                let ref cluster: HashMap<String, u32> = n_z_w[label];

                for word in doc {
                    lN2 += *cluster.get(word).unwrap_or(&0_u32) as f64 + beta;
                }
                for j in 0_u32..doc_size {
                    lD2 += (n_z[label] + j) as f64 - 1_f64 + V * beta;
                }
                lN2 = if lN2 > 0_f64 { lN2.ln() } else { 0_f64 };
                lD2 = if lD2 > 0_f64 { lD2.ln() } else { 0_f64 };
                p[label] = (lN1 - lD1 + lN2 - lD2).exp();
            }
            let pnorm: f64 = p.iter().sum();
            for label in 0_usize..K {
                p[label] = p[label] / pnorm;
            }
            // choose the next cluster randomly according to the computed probability
            let z_new: usize = random_choice().random_choice_f64(&labels, &p, 1)[0].clone();

            // transfer document to the new cluster
            if z_new != z_old {
                total_transfers += 1;
            }
            d_z[i] = z_new;
            m_z[z_new] += 1_u32;
            n_z[z_new] += doc_size;

            {
                let ref mut new_clust_words: HashMap<String, u32> = n_z_w[z_new];
                for word in doc {
                    if !new_clust_words.contains_key(word) {
                        new_clust_words.insert(word.clone(), 0_u32);
                    }
                        * new_clust_words.get_mut(word).unwrap() += 1_u32;
                }
            }
        }
        number_clusters = n_z_w.iter().map(|c| if c.len()>0 {1} else {0} ).sum();
        if (it % 100 == 0) {
            println!("Iteration {}: {} docs transferred with {} clusters populated.", it, total_transfers, number_clusters);
        }
    }

    // write the labels
    {
        let error_msg = format!("Could not write file!");
        let mut f = File::create(&args.arg_labelout).expect(&error_msg);
        f.write_all(d_z.iter().map(|label| label.to_string()).collect::<Vec<String>>().join("\n").as_bytes());
    }

    // write the cluster descriptions
    {
        let error_msg = format!("Could not write file!");
        let mut f = File::create(&args.arg_clusterout).expect(&error_msg);
        for k in 0..K {
            let ref word_dist = n_z_w[k];
            let mut line = k.to_string() + " ";
            let mut dist_counts:Vec<String> = word_dist.iter().map(|(a,b)| a.clone() + ":" + &b.clone().to_string() ).collect();
            dist_counts.sort();
            line += &dist_counts.join(" ");
            f.write((line+"\n").as_bytes());
        }
    }

    fn lines_from_file(filename: &str) -> Vec<String>
    {
        let error_msg = format!("Could not read file {}!", filename);
        let file = File::open(filename).expect(&error_msg);
        let buf = BufReader::new(file);
        buf.lines().map(|l| l.expect("Could not parse line!")).collect()
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
