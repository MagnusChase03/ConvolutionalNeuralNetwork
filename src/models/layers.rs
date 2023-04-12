pub trait Layer {

    fn forward(&mut self, input: &Vec<Vec<f64>>) -> Vec<Vec<f64>>;

}

pub struct PoolingLayer {

    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize)

}

impl PoolingLayer {

    pub fn new(kernel_size: (usize, usize), stride: (usize, usize), padding: (usize, usize)) -> Self {

        PoolingLayer {kernel_size, stride, padding}

    }

}

impl Layer for PoolingLayer {

    fn forward(&mut self, input: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {

        let output_width: usize = ((input[0].len() + (2 * self.padding.0) - self.kernel_size.0) / self.stride.0) + 1;
        let output_height: usize = ((input.len() + (2 * self.padding.1) - self.kernel_size.1) / self.stride.1) + 1;

        let mut output = vec![vec![0.0; output_width]; output_height];

        for oy in 0..output_height {

            for ox in 0..output_width {

                let mut max: f64 = std::f64::NEG_INFINITY;
                for ky in 0..self.kernel_size.1 {

                    for kx in 0..self.kernel_size.0 {

                        let y: usize = (self.stride.1 * oy) + ky;
                        let x: usize = (self.stride.0 * ox) + kx;

                        if y < self.padding.1 || x < self.padding.0
                            || y - self.padding.1 >= input.len() 
                            || x - self.padding.0 >= input[0].len() {

                            if 0.0 > max {

                                max = 0.0;

                            }

                        } else {

                            let value: f64 = input[y - self.padding.1][x - self.padding.0];
                            if value > max {

                                max = value;

                            }

                        }
                        
                    }


                }

                output[oy][ox] = max;

            }

        }

        output

    }

}