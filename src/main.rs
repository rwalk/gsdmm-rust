extern crate docopt;
extern crate rustc_serialize;
extern crate gsdmm;

use gsdmm::GSDMM;
use docopt::Docopt;
use std::io::{BufRead,BufReader};
use std::fs::File;
use std::collections::HashSet;
use std::io::Write;
use std::cmp::Ordering;

const USAGE: &'static str ="
Gibbs sampling algorithm for a Dirichlet Mixture Model of Yin and Wang 2014.

Usage:
  gsdmm <datafile> <vocabfile> <outprefix> [-k <max_clusters>] [-a <alpha>] [-b <beta>] [-m <maxit>]
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
    arg_outprefix: String,
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
        let mut term_vector = line.to_owned()
            .split_whitespace()
            .map(|s| s.to_owned())
            .filter(|s| (&vocab).contains(s))
            .collect::<Vec<String>>();

        // sort and dedupe: this implementation requires binary term counts
        term_vector.sort();
        term_vector.dedup();
        term_vector
    }).collect::<Vec<Vec<String>>>();

    let mut model = GSDMM::new(args.flag_alpha, args.flag_beta, args.flag_k, args.flag_maxit, vocab, docs);
    model.fit();

    // write the labels
    {
        let fname = (&args.arg_outprefix).clone() + "labels.csv";
        let error_msg = format ! ("Could not write file {}!", fname);
        let mut f = File::create( fname ).expect( & error_msg);
        let mut scored = Vec::<(String,String,String)>::new();

        // zip with the input data so we get clustered, raw input documents in the output set
        for (doc,txt) in (&model.docs).iter().zip(lines_from_file(&args.arg_datafile).iter()) {
            let p = model.score( & doc);
            let mut row = p.iter().enumerate().collect::<Vec<_>>();
            if row_has_nan(&row, &doc) {
                scored.push(("-1".to_string(), "0".to_string(), txt.clone()));
            } else {
                row.sort_by(|a, b| (a.1.partial_cmp(b.1)).unwrap());
                let line = row.pop().unwrap();
                scored.push((line.0.to_string(), line.1.to_string(), txt.clone()));
            }
        }
        scored.sort();
        for (label, score, txt) in scored {
            f.write((label + "," + &score + "," + &txt + "\n").as_bytes());
        }
    }

    // write the cluster descriptions
    {
        let fname = (&args.arg_outprefix).clone() + "cluster_descriptions.txt";
        let error_msg = format!("Could not write file {}!", fname);
        let mut f = File::create(fname).expect(&error_msg);
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

    fn row_has_nan(row:&Vec<(usize, &f64)>, doc:&Vec<String>) -> bool {
        for entry in row {
            if entry.1.is_nan() {
                println!("Cluster: {:?} has NaN score for document {:?}", entry, doc);
                return true
            }
        }
        return false;
    }
}
