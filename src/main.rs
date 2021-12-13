use ndarray::{arr2, Axis};
use std::ops::{AddAssign, Mul, SubAssign};

// User Based collaborative filtering
fn main() {
    let data_matrix = arr2(&[
        [5., 4., 1., 4., 0.],
        [3., 1., 2., 3., 3.],
        [4., 3., 4., 3., 5.],
        [3., 3., 1., 5., 4.],
    ]);
    let mut final_matrix = data_matrix.clone();

    // Remove the user with index 0
    let mut positions = Vec::new();
    for datum in final_matrix.rows() {
        let pos = datum
            .indexed_iter()
            .filter(|(_, &x)| x == 0.)
            .map(|(index, _)| index)
            .collect::<Vec<usize>>();
        positions.extend(pos);
    }

    for position in positions {
        final_matrix.remove_index(Axis(1), position);
    }

    // Find a ri and update the matrix
    let ncols = final_matrix.ncols();
    let mut ris = Vec::new();
    for datum in final_matrix.rows_mut() {
        let sum = datum.sum();
        let ri = sum / ncols as f64;
        ris.push(ri);
        for data in datum {
            data.sub_assign(ri);
        }
    }

    let predict_i = 0;
    let predict_j = 4;
    let predict_value = predict(final_matrix, predict_i, data_matrix, predict_j, ris);
    println!("predict_value = {}", predict_value);
}

fn predict(
    final_matrix: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>>,
    predict_i: usize,
    data_matrix: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>>,
    predict_j: usize,
    ris: Vec<f64>,
) -> f64 {
    let mut similarity_a = 0.;
    let mut similarity_b = 0.;
    for (index, _) in final_matrix.rows().into_iter().enumerate() {
        if index == predict_i {
            continue;
        }
        similarity_a.add_assign(
            similarity(&final_matrix, predict_i, index)
                .mul(data_matrix.get((index, predict_j)).unwrap() - ris[index]),
        );
        similarity_b.add_assign(similarity(&final_matrix, predict_i, index).abs());
    }
    ris[predict_i] + (similarity_a / similarity_b)
}

fn similarity(data_matrix: &ndarray::Array2<f64>, i: usize, j: usize) -> f64 {
    let mut a = 0.;
    let mut b_a = 0.;
    let mut b_b = 0.;
    for nc in 0..=data_matrix.ncols() - 1 {
        a += data_matrix.row(i)[nc].mul(data_matrix.row(j)[nc]);
        b_a += data_matrix.row(i)[nc].abs().powi(2);
        b_b += data_matrix.row(j)[nc].abs().powi(2);
    }
    a / (b_a.sqrt() * b_b.sqrt())
}
