// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.Recognizers.Text;
using Microsoft.Recognizers.Text.Choice;
using Microsoft.Recognizers.Text.DateTime;
using Microsoft.Recognizers.Text.Number;
using Microsoft.Recognizers.Text.NumberWithUnit;
using Microsoft.Recognizers.Text.Sequence;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AllData
{
    public static class Program
    {
        // Use English for the Recognizers culture
        private const string DefaultCulture = Culture.English;

        public static void Main(string[] args)
        {
            // Enable support for multiple encodings, especially in .NET Core
            Encoding.RegisterProvider(CodePagesEncodingProvider.Instance);
            var recognizer = new NumberRecognizer(Culture.English);
            var model = recognizer.GetNumberModel();
            string path = args.Length == 0 ? @"C:\Users\v-hongweihan\Downloads\NumberDataset\text.json" : args[0];
            var file = File.OpenText(path);
            var reader = new JsonTextReader(file);
            var arr = (JArray)JToken.ReadFrom(reader);
            var arr_out = new JObject();
            int id = 0;
            int total = arr.Count();
            foreach (var text in arr)
            {
                var res = Parse_string(model, text.ToString());
                if (res != null)
                {
                    arr_out[id.ToString()] = Parse_string(model, text.ToString());
                }

                id++;
                if (id % 10000 == 0)
                {
                    Console.WriteLine((1.0 * id / total).ToString());
                }

            }

            File.WriteAllText(path + ".num", JsonConvert.SerializeObject(arr_out, Formatting.Indented));
        }

        private static JArray Parse_string(NumberModel model, string text)
        {
            if (text.Length == 0)
            {
                return null;
            }

            var vals = model.Parse(text);
            if (vals.Count() == 0)
            {
                return null;
            }

            var ret = new JArray();
            foreach (var val in vals)
            {
                var obj = new JObject();
                obj["Text"] = val.Text;
                obj["Start"] = val.Start;
                obj["End"] = val.End;
                obj["TypeName"] = val.TypeName;
                var resolution = new JObject();
                resolution["value"] = val.Resolution["value"].ToString();
                obj["Resolution"] = resolution;
                ret.Add(obj);
            }

            return ret;
        }
    }
}
