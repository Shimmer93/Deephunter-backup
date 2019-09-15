import time
import numpy as np
from random import  randint
import tensorflow as tf
import datetime
import random
import os
class Seed(object):
    """Class representing a single element of a corpus."""

    def __init__(self, cl, coverage, root_seed, parent, metadata, ground_truth, l0_ref=0, linf_ref=0):
        """Inits the object.

        Args:
          cl: a transformation state to represent whether this seed has been
          coverage: a list to show the coverage
          root_seed: maintain the initial seed from which the current seed is sequentially mutated
          metadata: the prediction result
          ground_truth: the ground truth of the current seed

          l0_ref, linf_ref: if the current seed is mutated from affine transformation, we will record the l0, l_inf
          between initial image and the reference image. i.e., L0(s_0,s_{j-1}) L_inf(s_0,s_{j-1})  in Equation 2 of the paper
        Returns:
          Initialized object.
        """

        self.clss =  cl
        self.metadata = metadata
        self.parent = parent
        self.root_seed = root_seed
        self.coverage = coverage
        self.queue_time = None
        self.id = None
        # The initial probability to select the current seed.
        self.probability = 0.8
        self.fuzzed_time = 0

        self.ground_truth = ground_truth

        self.l0_ref = l0_ref
        self.linf_ref = linf_ref




class FuzzQueue(object):
    """Class that holds inputs and associated coverage."""

    def __init__(self, outdir, is_random, sample_type, cov_num, criteria):
        """Init the class.
        """

        self.plot_file = open(os.path.join(outdir, 'plot.log'), 'a+')
        self.out_dir = outdir
        self.mutations_processed = 0
        self.queue = []
        self.sample_type = sample_type
        self.start_time = time.time()
        # whether it is random testing
        self.random = is_random
        self.criteria = criteria

        self.log_time = time.time()
        # Like AFL, it records the coverage of the seeds in the queue
        self.virgin_bits = np.full(cov_num, 0xFF, dtype=np.uint8)
        # self.adv_bits = np.full(cov_num, 0xFF, dtype=np.uint8)

        self.uniq_crashes = 0
        self.total_queue = 0
        self.total_cov = cov_num


        # Some log information
        self.last_crash_time = self.start_time
        self.last_reg_time = self.start_time
        self.current_id = 0
        self.seed_attacked = set()
        self.seed_attacked_first_time = dict()



        self.dry_run_cov = None


        # REG_MIN and REG_GAMMA are the p_min and gamma in Equation 3
        self.REG_GAMMA = 5
        self.REG_MIN = 0.3
        self.REG_INIT_PROB = 0.8



    def has_new_bits(self, seed):

        temp = np.invert(seed.coverage, dtype = np.uint8)
        cur = np.bitwise_and(self.virgin_bits, temp)
        has_new = not np.array_equal(cur, self.virgin_bits)
        if has_new:
            # If the coverage is increased, we will update the coverage
            self.virgin_bits = cur
        return has_new or self.random

    def plot_log(self, id):
        # Plot the data during fuzzing, include: the current time, current iteration, length of queue, initial coverage,
        # total coverage, number of crashes, number of seeds that are attacked, number of mutations, mutation speed
        queue_len = len(self.queue)
        coverage = self.compute_cov()
        current_time = time.time()
        self.plot_file.write(
            "%d,%d,%d,%s,%s,%d,%d,%s,%s\n" %
            (time.time(),
             id,
             queue_len,
             self.dry_run_cov,
             coverage,
             self.uniq_crashes,
             len(self.seed_attacked),
             self.mutations_processed,
             round(float(self.mutations_processed) / (current_time - self.start_time), 2)
             ))
        self.plot_file.flush()
    def write_logs(self):
        log_file = open(os.path.join(self.out_dir, 'fuzz.log'), 'w+')
        for k in self.seed_attacked_first_time:
            log_file.write("%s:%s\n"%(k, self.seed_attacked_first_time[k]))
        log_file.close()
        self.plot_file.close()


    def log(self):
        queue_len = len(self.queue)
        coverage = self.compute_cov()
        current_time = time.time()
        tf.logging.info(
                "Metrics %s | corpus_size %s | crashes_size %s | mutations_per_second: %s | total_exces %s | last new reg: %s | last new adv %s | coverage: %s -> %s%%",
                self.criteria,
                queue_len,
                self.uniq_crashes,
                round(float(self.mutations_processed)/(current_time - self.start_time), 2),
                self.mutations_processed,
                datetime.timedelta(seconds=(time.time() - self.last_reg_time)),
                datetime.timedelta(seconds=(time.time() - self.last_crash_time)),
                self.dry_run_cov,
                coverage
            )
    def compute_cov(self):
        # Compute the current coverage in the queue
        coverage = round(float(self.total_cov - np.count_nonzero(self.virgin_bits == 0xFF)) * 100 / self.total_cov, 2)
        return str(coverage)

    def tensorfuzz(self):
        """Grabs new input from corpus according to sample_function."""
        # choice = self.sample_function(self)
        corpus = self.queue
        reservoir = corpus[-5:] + [random.choice(corpus)]
        choice = random.choice(reservoir)
        return choice
        # return random.choice(self.queue)

    def select_next(self):
        # Different seed selection strategies (See details in Section 4)
        if self.random == 1 or self.sample_type == 'uniform':
            return self.random_select()
        elif self.sample_type == 'tensorfuzz':
            return self.tensorfuzz()
        elif self.sample_type == 'deeptest':
            return self.deeptest_next()
        elif self.sample_type == 'prob':
            return self.prob_next()
    def random_select(self):
        return random.choice(self.queue)

    def deeptest_next(self):
        choice = self.queue[-1]
        return choice

    def fuzzer_handler(self, iteration, cur_seed, bug_found, coverage_inc):
        # The handler after each iteration
        if self.sample_type == 'deeptest' and not coverage_inc:
            # If deeptest cannot increase the coverage, it will pop the last seed from the queue
            self.queue.pop()

        elif self.sample_type == 'prob':
            # Update the probability based on the Equation 3 in the paper
            if cur_seed.probability > self.REG_MIN and cur_seed.fuzzed_time < self.REG_GAMMA * (1 - self.REG_MIN):
                cur_seed.probability = self.REG_INIT_PROB - float(cur_seed.fuzzed_time) / self.REG_GAMMA

        if bug_found:
            # Log the initial seed from which we found the adversarial. It is for the statics of Table 6
            self.seed_attacked.add(cur_seed.root_seed)
            if not (cur_seed.parent in self.seed_attacked_first_time):
                # Log the information about when (which iteration) the initial seed is attacked successfully.
                self.seed_attacked_first_time[cur_seed.root_seed] = iteration



    def prob_next(self):
        """Grabs new input from corpus according to sample_function."""
        while True:
            if self.current_id == len(self.queue):
                self.current_id = 0

            cur_seed = self.queue[self.current_id]
            if randint(0,100) < cur_seed.probability * 100:
                # Based on the probability, we decide whether to select the current seed.
                cur_seed.fuzzed_time += 1
                self.current_id += 1
                return cur_seed
            else:
                self.current_id += 1
