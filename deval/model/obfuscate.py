import subprocess
import os

class Obfuscator:

    @classmethod
    def obfuscate(source_dir: str):
        """
        Obfuscates the Python code in the given directory using PyArmor.
        The output is stored in a folder called 'obfuscated_pipeline'.

        Args:
            source_dir (str): The path to the directory containing the code to be obfuscated.
        """
        if not os.path.exists(source_dir):
            raise ValueError(f"Source directory '{source_dir}' does not exist.")
        
        # Define the output directory for the obfuscated code
        output_dir = os.path.join(source_dir, 'obfuscated_pipeline')

        # Create the output directory if it does not exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        try:
            # Run the PyArmor obfuscation command
            subprocess.run([
                'pyarmor', '-O', output_dir, source_dir
            ], check=True)
            print(f"Code successfully obfuscated and saved to: {output_dir}")
        
        except subprocess.CalledProcessError as e:
            print(f"PyArmor obfuscation failed: {e}")
            raise




if __name__ == "__main__":
    from huggingface.pipeline import DeValPipeline

    # Now use DeValPipeline as normal
    model_dir = "../model"

    tasks = ['hallucination']
    rag_context = "The earth is round. The sky is Blue."
    llm_response = "The earth is flat."
    query = ""

    pipe = DeValPipeline("de_val", model_dir = model_dir)
    print(pipe("", tasks=tasks, rag_context=rag_context, query=query, llm_response=llm_response))
