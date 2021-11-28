using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography.X509Certificates;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using Tensor = System.Numerics.Tensors.Tensor;

namespace WFInfo_onnx
{
    class Program
    {
        public static Mat _hold;
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
            var image = LoadImage("testfiles/test3.png");
            var result = Run(image);
            var parsed = ParseResults(result);
            Console.WriteLine(result);
        }
        public static IEnumerable<float> Run(DenseTensor<float> inputTensor)
        {

            // Get path to model to create inference session.
            var modelPath = "models/recognizer.onnx";

            // create input tensor (nlp example)
            // var inputTensor = new DenseTensor<string>(new string[] { "" }, new int[] { 1, 1 });

            // Create input data for session.
            var input = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<float>("input", inputTensor) };

            // Create an InferenceSession from the Model Path.
            var session = new InferenceSession(modelPath);

            // Run session and send input data in to get inference output. Call ToList then get the Last item. Then use the AsEnumerable extension method to return the Value result as an Enumerable of NamedOnnxValue.
            var outputRaw = session.Run(input);
            var output = outputRaw.ToList().Last();

            // From the Enumerable output create the inferenceResult by getting the First value and using the AsDictionary extension method of the NamedOnnxValue.
            var inferenceResult = output.AsEnumerable<float>();

            // Return the inference result as json.
            return inferenceResult;
        }

        public static string ParseResults(IEnumerable<float> output)
        {
            //chunk output into pieces of 97
            var chunks = output.Select(Convert.ToDouble).Chunk(97);
            var softmaxed = chunks.Select(SoftMax);
            var normalised = softmaxed.Select(doubles => doubles.Select(y => y / doubles.Sum()));
            var indexed = normalised.Select(row => row.ToList().IndexOf(row.Max()));
            return GreedyParse(indexed, 97);
            // return "";
        }

        public static string GreedyParse(IEnumerable<int> input, int length)
        {
            var text_index = input.ToArray();
            var rest =
                "0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ €ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                    .Select(c => c.ToString());
            var characters = rest.Prepend("[blank]");
            var ignore_idx = new int[] { 0 };
            var texts = new List<string>();
            var index = 0;
            for (int l = 0; l < length; l++)
            {
                var t = text_index[index..(index + 1)];
            }

            return "";

        }
        public static IEnumerable<double> SoftMax(IEnumerable<double> input)
        {
            var z_exp = input.Select(Math.Exp);
            // [2.72, 7.39, 20.09, 54.6, 2.72, 7.39, 20.09]

            var sum_z_exp = z_exp.Sum();
            // 114.98

            var softmax = z_exp.Select(i => i / sum_z_exp);
            // [0.024, 0.064, 0.175, 0.475, 0.024, 0.064, 0.175]

            return softmax;

        }

        public static Mat ResizeImage(Mat image, int imgH, int imgW)
        {
            var w = image.Width;
            var h = image.Height;

            var ratio = w / (double)h;
            int resizedW = 0;
            if (Math.Ceiling(imgH * ratio) > imgW)
            {
                resizedW = imgW;
            }
            else
            {
                resizedW = (int)Math.Ceiling(imgH * ratio);
            }

            var resized_image = image.Resize(new Size(resizedW, imgH), interpolation: InterpolationFlags.Cubic);
            return resized_image;
        }

        public static Mat NormalizePad(Mat image, int imgH, int imgW)
        {
            // var image3 = new Mat();
            // image.ConvertTo(image3, MatType.CV_32F);
            // image.SaveImage("outa.png");
            var image4 = image.Divide(255.0).ToMat();
            // image4.Subtract(0.5);
            // var tempOut = new Mat();
            // var temp = image4.Subtract(0.5).Multiply(255).ToMat().CopyTo(tempOut, MatType.CV_8UC1);
            // Cv2.ImWrite("outy.png", tempOut);
            // image4.Subtract(0.5).Multiply(255.0).ToMat().SaveImage("outx.png");
            // var image5 = image4.Subtract(0.5).Divide(0.5).ToMat();
            var image5 = image4.Subtract(0.5).Divide(0.5).ToMat();
            
            // var tempOut = new Mat();
            // image5.Multiply(255.0).ToMat().ConvertTo(tempOut, MatType.CV_8UC1);
            // Cv2.ImWrite("outfinal2.png", tempOut);
            // image5.Multiply(255.0).ToMat().SaveImage("out5.png");
            
            return image5.CopyMakeBorder(0, 0, 0, imgW - image5.Width, BorderTypes.Replicate);
        }
        public static DenseTensor<float> LoadImage(string path)
        {
            var image = OpenCvSharp.Cv2.ImRead(path, ImreadModes.Grayscale);

            var targetHeight = 64;
            var maxWidth = 300;
            // var image2 = new Mat();
            // image.ConvertTo(image2, MatType.CV_8S);

            var image2 = ResizeImage(image, targetHeight, maxWidth);

            image.SaveImage("out1.png");
            image2.SaveImage("out2.png");
            // image = image.Subtract(0.5).Divide(0.5);
            // image = image.Resize(new Size(maxWidth, targetHeight), interpolation: InterpolationFlags.Linear);
            var image3 = new Mat();
            image2.ConvertTo(image3, MatType.CV_32FC1);
            // image3.SaveImage("out3.png");
            // var image4 = image3.Divide(255).Subtract(0.5).Divide(0.5).ToMat();
            var image4 = NormalizePad(image3,targetHeight, maxWidth);

            // var tempOut = new Mat();
            // image4.Multiply(255).ToMat().ConvertTo(tempOut, MatType.CV_8U);
            // Cv2.ImWrite("outfinal.png", tempOut);
            _hold = image4; 
            // var tempOut = image4.Multiply(255).ToMat();
            // tempOut.SaveImage("testout.png");
            var width = image4.Width;
            var height = image4.Height;
            // var width = maxWidth;
            // var height = targetHeight;
            var data = new DenseTensor<float>(new[] { 1, 1, height, width});

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    data[0, 0, y, x] = image4.Get<float>(y, x);
                }
            }
            //
            // for (int y = 0; y < height; y++)
            // {
            //     for (int x = 0; x < width; x++)
            //     {
            //         data[0, 0, x, y] = ((data[0, 0, x, y] / 255.0f) - 0.5f) / 0.5f;
            //     }
            //     
            // }
           
            
            
            

            return data;

        }
 
    }
}