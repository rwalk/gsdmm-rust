extern crate docopt;
extern crate rustc_serialize;
extern crate gsdmm;

use gsdmm::GSDMM;
use docopt::Docopt;
use std::io::{BufRead,BufReader};
use std::fs::File;
use std::collections::{HashSet, HashMap};
use std::io::Write;

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
    flag_k: usize,
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

    let mut model = GSDMM::new(args.flag_alpha, args.flag_beta, args.flag_k, args.flag_maxit, vocab, docs);
    model.fit();

    // write the label probabilities
    let error_msg = format!("Could not write file!");
    let mut f = File::create(&args.arg_labelout).expect(&error_msg);
    for doc in &(model.docs) {
        let p = model.score(&doc);
        let line = p.iter().map(|k| k.to_string()).collect::<Vec<String>>().join(",");
        f.write((line+"\n").as_bytes());
    }

    // write the cluster descriptions
    {
        let error_msg = format!("Could not write file!");
        let mut f = File::create(&args.arg_clusterout).expect(&error_msg);
        for k in 0..args.flag_k {
            let ref word_dist = model.cluster_word_distributions[k];
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

}
