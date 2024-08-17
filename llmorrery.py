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
import uuid
import yaml
import ollama
from dotenv import load_dotenv
from openai import OpenAI
from flask import Flask, request, Response, send_from_directory, redirect, session


class LLMOrrery:
    """
        LLMOrrery run your software using a LLM as an interpreter.
    """

    def __init__(self, arguments: argparse.Namespace) -> None:
        # Setup Sesssions
        self.session: dict[str, dict] = {}
        # Get Arguments
        self.args = arguments
        # load the config file
        self._load_config()
        # Update Model if given in the arguemnts
        if arguments.model is not None:
            self.config["model"]["model"] = arguments.model
        # print Model to use
        print(f"Model to use: {self.config['model']['model']}")
        # setup model object
        self.model = self._get_model_object(self.config["model"]["model"])
        # setup flask app
        self.app = Flask(__name__)
        # Set secret Key
        self.app.secret_key = os.urandom(24)
        # add routes
        self._register_routes()

    def _setup_logfile(self) -> str:
        """
        Gives the logfile path back based on the arguments given to the programm
        """
        filename = os.path.basename(self.args.configfile)[
            :-5]+"_"+time.strftime("%Y%m%d-%H%M%S")+".yaml"
        if self.args.logpath is not None:
            return str(os.path.join(args.logpath, filename))
        return str(os.path.join(os.path.dirname(self.args.configfile), filename))

    def _load_config(self) -> None:
        """
        Loads the configuration from the given configfile
        """
        with open(self.args.configfile, "r", encoding="utf-8") as fin:
            self.config = yaml.safe_load(fin)

    def _setup_prompts(self) -> list[dict[str, str]]:
        """
        Sets up the prompts for the LLM
        """
        prompts = []
        for row in self.config["prompt"]:
            key = list(row.keys())[0]
            if key == "system":
                row[key] += (
                    "\nReason first on what the user expects in the site to see and than generate the HTML "
                    "code for that site. When you have made you analysis, write the HTML code, "
                    "by starting with <!DOCTYPE html> and ending with </html>."
                )
            prompts.append({"role": key, "content": row[key]})
        return prompts

    def _register_routes(self) -> None:
        """
        Registers the routes for the LLM
        """
        self.app.route("/favicon.ico")(self.favicon)
        self.app.route("/reset")(self.reset)
        self.app.route("/", defaults={"path": ""},
                       methods=["GET"])(self.catch_all_get)
        self.app.route("/<path:path>", methods=["GET"])(self.catch_all_get)

    @staticmethod
    def _clean_html_response(response: str) -> str:
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
        """
        Returns the appropiate model object from the model name
        """
        if model_name.startswith("ollama:"):
            # ollama model
            model_path = model_name.split(":", maxsplit=1)[1]
            client_ollama = ollama.Client()
            return lambda x: client_ollama.chat(model=model_path, messages=x, stream=False)["message"]["content"]
        if model_name.startswith("openai:"):
            # openai model
            model_path = model_name.split(":", maxsplit=1)[1]
            client_openai = OpenAI()
            return lambda x: client_openai.chat.completions.create(model=model_path, messages=x, stream=False).choices[0].message.content
        raise Exception(f"Unknown model type {model_name}")

    def _export_to_file(self, user_id: str) -> None:
        """
        Exports the session to a file
        """
        with open(self.session[user_id]["filename"], "w", encoding="utf-8") as fout:
            export = {
                "model": self.config["model"],
                "description": self.config["description"],
                "prompt": [{item["role"]: item["content"]} for item in self.session[user_id]["prompts"]],
            }
            yaml.dump(export, fout)

    def catch_all_get(self, path: str):
        """
        Catches all GET requests and generates a response based on the request
        """

        user_id = self._session_handling()

        # get user prompt from request
        user_prompt = f"{request.remote_addr}:\nGET {request.full_path} HTTP/1.1\n"
        self.session[user_id]["prompts"].append(
            {"role": "user", "content": user_prompt})

        # get response from LLM
        response = self.model(self.session[user_id]["prompts"])

        # add response to prompts
        self.session[user_id]["prompts"].append(
            {"role": "assistant", "content": response}
        )

        # clean the HTML response from the LLM
        clean_response = self._clean_html_response(response)

        # export the conversation to a file
        self._export_to_file(user_id)

        # return the cleaned response
        return Response(clean_response, mimetype="text/html")

    def reset(self):
        """
            Resets the current conversation by clearing the prompts and reloading the initial prompt from the config file.
        """
        user_id = self._session_handling()
        self.session[user_id] = {"prompts": self._setup_prompts(),
                                 "filename": self._setup_logfile()}
        print(f"Reset\nNew Log File name: {self.session[user_id]['filename']}")
        # Send a redirect to the main page
        return redirect("/")

    def favicon(self):
        """
            Return image favicon.ico, to make browsers happy
        """
        # Return image favicon.ico
        return send_from_directory(os.path.join(self.app.root_path, "."), "img/favicon.ico", mimetype="image/vnd.microsoft.icon")

    def run(self) -> None:
        """
        Runs the Flask app
        """
        self.app.run(host=self.args.host, port=self.args.port, threaded=False)

    def _session_handling(self) -> str:
        """
        Handles session management for the application.
        """
        if self.config["model"].get("sessiontype", "global") == "global":
            user_id = "global"
        elif self.config["model"].get("sessiontype", "global") == "user":
            if "user_id" not in session:
                # If it's a new session, generate a unique ID and store it in the session
                user_id = str(uuid.uuid4())
                session["user_id"] = user_id
            else:
                # If it's an existing session, retrieve the user ID from the session
                user_id = session["user_id"]
        else:
            # Throw value Error
            raise ValueError(
                f"Invalid session type: {self.config['model']['sessiontype']}")

        if user_id not in self.session:
            print("Setup User Session")
            self.session[user_id] = {"prompts": self._setup_prompts(),
                                     "filename": self._setup_logfile()}
        return user_id


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
