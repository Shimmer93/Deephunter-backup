import tensorflow as tf
import gc
import time
class Fuzzer(object):
    """Class representing the fuzzer itself."""

    def __init__(
        self,
        corpus,
        coverage_function,
        metadata_function,
        objective_function,
        mutation_function,
        fetch_function,
        iterate_function,
        plot = True
    ):
        """Init the class.

    Args:
      corpus: An InputCorpus object.
      coverage_function: a function that does CorpusElement -> Coverage.
      metadata_function: a function that does CorpusElement -> Metadata.
      objective_function: a function that checks if a CorpusElement satisifies
        the fuzzing objective (e.g. find a NaN, find a misclassification, etc).
      mutation_function: a function that does CorpusElement -> Metadata.
      fetch_function: grabs numpy arrays from the TF runtime using the relevant
        tensors.
    Returns:
      Initialized object.
    """
        self.plot = plot
        self.queue = corpus
        self.coverage_function = coverage_function
        self.metadata_function = metadata_function
        self.objective_function = objective_function
        self.mutation_function = mutation_function
        self.fetch_function = fetch_function
        self.iterate_function = iterate_function

    def loop(self, iterations):
        """Fuzzes a machine learning model in a loop, making *iterations* steps."""
        iteration = 0
        while True:

            if len(self.queue.queue) < 1 or iteration >= iterations:
                break
            if iteration % 100 == 0:
                tf.logging.info("fuzzing iteration: %s", iteration)
                gc.collect()


            parent = self.queue.select_next()
            # Get a mutated batch for each input
            mutated_data_batches = self.mutation_function(parent)
            # Grab the coverage and metadata for mutated batch
            coverage_batches, metadata_batches = self.fetch_function(
                mutated_data_batches
            )

            # Plot the data
            if self.plot:
                self.queue.plot_log(iteration)


            if coverage_batches is not None and len(coverage_batches) > 0:
                # Get the coverage - one from each batch element
                mutated_coverage_list = self.coverage_function(coverage_batches)

                # Get the metadata objects - one from each batch element
                mutated_metadata_list = self.metadata_function(metadata_batches)

                # Check for each mutant and decide whether it will be saved
                bug_found, cov_inc = self.iterate_function(self.queue, parent.root_seed, parent, mutated_coverage_list, mutated_data_batches,mutated_metadata_list, self.objective_function)
                del mutated_coverage_list
                del mutated_metadata_list
            else:
                bug_found = False
                cov_inc = False



            self.queue.fuzzer_handler(iteration, parent, bug_found, cov_inc)
            iteration += 1

            del mutated_data_batches
            del coverage_batches
            del metadata_batches
        return None
