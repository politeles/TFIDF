using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using XGBoost;

namespace TFIDFExample
{
    class Program
    {
        static void Main(string[] args)
        {
            // Some example email body.
            string[] email =
            {
                "I have a problem with my payroll",
                "I'm gonna leave this company.",
                "I'm going on holidays",
                "I can't see my payroll",
            };

            // let's say this is the classification for each email:
            float[] category = { 0, 1, 1, 0 };
            // category 0 => payroll
            // category 1 => other stuff


            // Apply TF*IDF to the documents and get the resulting vectors.
            // This is what I call train on TFIDF:
            double[][] inputs = TFIDF.Transform(email, 0);
            inputs = TFIDF.Normalize(inputs);

            // Training for XGBoost (the classifier):
            var classifier = new XGBClassifier();
            classifier.Fit(inputs, category);


            // Once is trained, we want to classify a new text:
            string[] new_email =
                { "Hey, I didn't get any money this month, can you check what's going on with my payroll?"};
            // then, we have to apply the TFIDF, based on the documents we already have
            double[][] new_inputs = TFIDF.Transform(new_email, 0);
            new_inputs = TFIDF.Normalize(new_inputs);

            // predict!
            float predicted_class = classifier.Predict(new_inputs);

            Console.WriteLine("Press any key ..");
            Console.ReadKey();
        }
    }
}
