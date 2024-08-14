"""
LLMOrrery - PoC for running software using a LLM as an interpreter.

Copyright (C) 2024 Miklos Peuten

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU Affero General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along
with this program. If not, see <https://www.gnu.org/licenses/>.
"""

# includes
import os
import argparse
import time
import yaml
import ollama
from dotenv import load_dotenv
from openai import OpenAI
from flask import Flask, request, Response, send_from_directory, redirect


class LLMOrrery:
    """
        LLMOrrery run your software using a LLM as an interpreter.
    """

    def __init__(self, arguments) -> None:
        self.args = arguments
        # Setup Log Filename
        self._setup_logfile()
        print(f"Log File name: {self.filename}")
        # load the config file
        self._load_config()
        # Update Model if given in the arguemnts
        if arguments.model is not None:
            self.config["model"]["model"] = arguments.model
        # print Model to use
        print(f"Model to use: {self.config['model']['model']}")
        # setup prompts
        self._setup_prompts()
        # seup flask app
        self.app = Flask(__name__)
        # add routes
        self._register_routes()
        # get model object
        self.model = self._get_model_object(self.config["model"]["model"])

    def _setup_logfile(self):
        # Set logfilename if given use the path from the arguemnts else use the same
        # locaion as the config file
        # First filename component:
        filename = os.path.basename(self.args.configfile)[
            :-5]+"_"+time.strftime("%Y%m%d-%H%M%S")+".yaml"
        if self.args.logpath is not None:
            self.filename = os.path.join(args.logpath, filename)
        else:
            self.filename = os.path.join(
                os.path.dirname(self.args.configfile), filename)
        return self.filename

    def _load_config(self):
        """
        Loads the configuration from the given configfile
        """
        with open(self.args.configfile, "r", encoding="utf-8") as fin:
            self.config = yaml.safe_load(fin)

    def _setup_prompts(self):
        """
        Sets up the prompts for the LLM
        """
        self.prompts = []
        for row in self.config["prompt"]:
            key = list(row.keys())[0]
            if key == "system":
                row[key] += (
                    "\nReson first on what the user expects in the site to see and than generate the HTML "
                    "code for that site. When you have made you analysis, write the HTML code, "
                    "by starting with <!DOCTYPE html> and ending with </html>."
                )
            self.prompts.append({"role": key, "content": row[key]})

    def _register_routes(self):
        """
        Registers the routes for the LLM
        """
        self.app.route("/favicon.ico")(self.favicon)
        self.app.route("/reset")(self.reset)
        self.app.route("/", defaults={"path": ""},
                       methods=["GET"])(self.catch_all_get)
        self.app.route("/<path:path>", methods=["GET"])(self.catch_all_get)

    @staticmethod
    def _clean_html_response(response):
        """
        Cleans the HTML response from the LLM
        """
        clean_response = (
            response
            .replace("```html", "")
            .replace("```", "")
            .split("</html>", maxsplit=1)[0]
            + "</html>\n\n"
        )
        if "<!DOCTYPE html>" in clean_response:
            clean_response = (
                "<!DOCTYPE html>"
                + clean_response.split("<!DOCTYPE html>", maxsplit=1)[-1]
            )
        elif "<html" in clean_response:
            clean_response = clean_response.split(
                "<html", maxsplit=1)[-1] + "<html"
        else:
            # When there is no HTML in the output
            clean_response = "<html><head><title>No Output</title></head><body><h1>No Output</h1></body></html>"
        return clean_response

    @staticmethod
    def _get_model_object(model_name: str):
        if model_name.startswith("ollama:"):
            # ollama model
            model_path = model_name.split(":", maxsplit=1)[1]
            client = ollama.Client()
            return lambda x: client.chat(model=model_path, messages=x, stream=False)["message"]["content"]
        if model_name.startswith("openai:"):
            # openai model
            model_path = model_name.split(":", maxsplit=1)[1]
            client = OpenAI()
            return lambda x: client.chat.completions.create(model=model_path, messages=x, stream=False).choices[0].message.content
        raise Exception(f"Unknown model type {model_name}")

    def _export_to_file(self):
        with open(self.filename, "w", encoding="utf-8") as fout:
            export = {
                "model": self.config["model"],
                "description": self.config["description"],
                "prompt": [{item["role"]: item["content"]} for item in self.prompts],
            }
            yaml.dump(export, fout)

    def catch_all_get(self, path):
        """
        Catches all GET requests and generates a response based on the request
        """
        # get user prompt from request
        user_prompt = f"{request.remote_addr}:\nGET {request.full_path} HTTP/1.1\n"
        self.prompts.append({"role": "user", "content": user_prompt})

        # get response from LLM
        response = self.model(self.prompts)

        # add response to prompts
        self.prompts.append(
            {"role": "assistant", "content": response}
        )

        # clean the HTML response from the LLM
        clean_response = self._clean_html_response(response)

        # export the conversation to a file
        self._export_to_file()

        # return the cleaned response
        return Response(clean_response, mimetype="text/html")

    def reset(self):
        """
            Resets the current conversation by clearing the prompts and reloading the initial prompt from the config file.
        """
        self._setup_prompts()
        self._setup_logfile()
        print(f"Reset\nNew Log File name: {self.filename}")
        # Send a redirect to the main page
        return redirect("/")

    def favicon(self):
        """
            Return image favicon.ico, to make browsers happy
        """
        # Return image favicon.ico
        return send_from_directory(os.path.join(self.app.root_path, "."), "img/favicon.ico", mimetype="image/vnd.microsoft.icon")

    def run(self):
        """
        Runs the Flask app
        """
        self.app.run(host=self.args.host, port=self.args.port, threaded=False)


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="LLMOrrery")
    parser.add_argument(
        "-c",
        "--config",
        dest="configfile",
        help="The configuration file",
        required=True,
    )
    parser.add_argument(
        "-H",
        "--host",
        dest="host",
        type=str,
        help="The host to listen on",
        default="127.0.0.1",
    )
    parser.add_argument(
        "-p",
        "--port",
        dest="port",
        type=int,
        help="The port to listen on",
        default=8080,
    )
    parser.add_argument(
        "-m",
        "--model",
        dest="model",
        type=str,
        help="LLM model to use for this session.",
        default=None
    )
    parser.add_argument(
        "-l",
        "--logpath",
        dest="logpath",
        type=str,
        help="Path for the logfile, defaults to same path as the config file",
        default=None
    )
    args = parser.parse_args()

    interpreter_llm = LLMOrrery(args)

    interpreter_llm.run()
