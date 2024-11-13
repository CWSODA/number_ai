use faer::Mat;
use itertools::Itertools;

pub type Data = Mat<f32>;

pub type Label = u8;

pub fn load_data(img_file: &str, label_file: &str, lim: Option<usize>) -> Vec<(Data, Label)> {
    let mut output: Vec<(Data, Label)> = vec![];

    // for img, first 32 * 4 bits are not needed --> skip 16
    // for label, first 32 * 2 bits are not needed --> skip 8
    let img_u8 = std::fs::read(img_file).expect("Unable to open image file");
    let label_u8 = std::fs::read(label_file).expect("Unable to open label file");

    // if limit is set, read specific
    if let Some(lim) = lim {
        for (mut img_chunk, label) in img_u8
            .iter()
            .skip(16)
            .chunks(28 * 28)
            .into_iter()
            .zip(label_u8.iter().skip(8))
        {
            output.push((
                Mat::from_fn(28 * 28, 1, |_, _| *img_chunk.next().unwrap() as f32),
                *label,
            ));

            if output.len() >= lim {
                break;
            }
        }

        return output;
    }

    // else read all
    for (mut img_chunk, label) in img_u8
        .iter()
        .skip(16)
        .chunks(28 * 28)
        .into_iter()
        .zip(label_u8.iter().skip(8))
    {
        output.push((
            Mat::from_fn(28 * 28, 1, |_, _| *img_chunk.next().unwrap() as f32),
            *label,
        ));
    }

    output
}

pub fn print_image(data: &(Data, Label)) {
    let mut counter = 0;
    println!("Image of a {}:", data.1);

    // img should be a (28*28)x1 matrix so only column 0 is taken
    for e in data.0.col(0).iter() {
        if *e > 50.0 {
            print!("*");
        } else {
            print!(" ");
        }
        counter += 1;

        if counter % 28 == 0 {
            println!();
        }
    }

    println!("end image.")
}

pub fn label_to_target(label: Label) -> Mat<f32> {
    assert!(label <= 9, "Label {} is out of bounds.", label);

    // create matrix with zeros in all digits
    let mut output: Mat<f32> = Mat::zeros(10, 1);

    // set corresponding target digit to 1.0
    *(output.get_mut(label as usize, 0)) = 1.0;

    output
}
