"""Comprehensive failing tests for GMN parser mathematical expressions support.

This test suite defines expected behavior for parsing and evaluating mathematical expressions in GMN:
- Free energy calculations
- Probability distributions and operations
- Matrix operations and transformations
- Precision parameters and uncertainty quantification
- Bayesian inference operations
- Information theoretic measures

Following strict TDD: These tests MUST fail initially and drive implementation.
NO graceful fallbacks or try-except blocks allowed.
"""

from inference.active.gmn_parser import GMNParser, GMNSchemaValidator


class TestFreeEnergyCalculations:
    """Test parsing and evaluation of free energy mathematical expressions."""

    def test_parse_expected_free_energy_expression(self):
        """Test parsing expected free energy calculation expressions."""
        parser = GMNParser()

        spec = {
            "nodes": [
                {
                    "name": "policy_evaluation",
                    "type": "policy",
                    "expected_free_energy": {
                        "expression": "epistemic_value + pragmatic_value",
                        "epistemic_value": {
                            "formula": "-sum(q_pi * log(q_pi))",
                            "variables": {
                                "q_pi": "posterior_beliefs_over_states"
                            },
                        },
                        "pragmatic_value": {
                            "formula": "sum(q_pi * log_preferences)",
                            "variables": {
                                "q_pi": "posterior_beliefs_over_states",
                                "log_preferences": "log(C_vector)",
                            },
                        },
                        "normalization": "temperature_scaling",
                        "temperature": 1.0,
                    },
                }
            ],
            "edges": [],
        }

        result = parser.parse(spec)

        policy_node = result["nodes"][0]
        efe = policy_node["expected_free_energy"]

        assert efe["expression"] == "epistemic_value + pragmatic_value"
        assert "epistemic_value" in efe
        assert "pragmatic_value" in efe

        # Epistemic value (information gain)
        epistemic = efe["epistemic_value"]
        assert epistemic["formula"] == "-sum(q_pi * log(q_pi))"
        assert "q_pi" in epistemic["variables"]

        # Pragmatic value (goal achievement)
        pragmatic = efe["pragmatic_value"]
        assert pragmatic["formula"] == "sum(q_pi * log_preferences)"
        assert "log_preferences" in pragmatic["variables"]

        # Temperature scaling
        assert efe["normalization"] == "temperature_scaling"
        assert efe["temperature"] == 1.0

    def test_parse_variational_free_energy_expression(self):
        """Test parsing variational free energy (VFE) expressions."""
        parser = GMNParser()

        spec = {
            "nodes": [
                {
                    "name": "belief_update",
                    "type": "belief",
                    "variational_free_energy": {
                        "expression": "accuracy - complexity",
                        "accuracy": {
                            "formula": "sum(q_s * log(P_o_given_s))",
                            "variables": {
                                "q_s": "belief_over_states",
                                "P_o_given_s": "likelihood_matrix",
                            },
                        },
                        "complexity": {
                            "formula": "KL(q_s || p_s)",
                            "variables": {
                                "q_s": "posterior_beliefs",
                                "p_s": "prior_beliefs",
                            },
                            "kl_divergence": {
                                "type": "categorical",
                                "regularization": 1e-8,
                            },
                        },
                        "minimization": {
                            "method": "variational_message_passing",
                            "iterations": 16,
                            "convergence_threshold": 1e-6,
                        },
                    },
                }
            ],
            "edges": [],
        }

        result = parser.parse(spec)

        belief_node = result["nodes"][0]
        vfe = belief_node["variational_free_energy"]

        assert vfe["expression"] == "accuracy - complexity"

        # Accuracy term (expected log likelihood)
        accuracy = vfe["accuracy"]
        assert accuracy["formula"] == "sum(q_s * log(P_o_given_s))"
        assert "q_s" in accuracy["variables"]
        assert "P_o_given_s" in accuracy["variables"]

        # Complexity term (KL divergence)
        complexity = vfe["complexity"]
        assert complexity["formula"] == "KL(q_s || p_s)"
        kl_params = complexity["kl_divergence"]
        assert kl_params["type"] == "categorical"
        assert kl_params["regularization"] == 1e-8

        # Minimization parameters
        minimization = vfe["minimization"]
        assert minimization["method"] == "variational_message_passing"
        assert minimization["iterations"] == 16

    def test_parse_conditional_free_energy_with_policies(self):
        """Test parsing conditional free energy expressions with policy dependencies."""
        parser = GMNParser()

        spec = {
            "nodes": [
                {
                    "name": "policy_prior",
                    "type": "policy",
                    "conditional_free_energy": {
                        "expression": "sum_tau(G_tau * gamma^tau)",
                        "variables": {
                            "G_tau": {
                                "formula": "expected_free_energy_at_time_tau",
                                "computation": {
                                    "epistemic": "-sum(q_s_tau * H(A_matrix))",
                                    "pragmatic": "sum(q_o_tau * log(C_tau))",
                                },
                            },
                            "gamma": "temporal_discount_factor",
                            "tau": "time_index",
                        },
                        "temporal_horizon": 5,
                        "policy_precision": {
                            "beta": 2.0,
                            "adaptive": True,
                            "update_rule": "exponential_moving_average",
                        },
                    },
                }
            ],
            "edges": [],
        }

        result = parser.parse(spec)

        policy_node = result["nodes"][0]
        cfe = policy_node["conditional_free_energy"]

        assert cfe["expression"] == "sum_tau(G_tau * gamma^tau)"
        assert cfe["temporal_horizon"] == 5

        # Expected free energy components
        G_tau = cfe["variables"]["G_tau"]
        assert "computation" in G_tau
        computation = G_tau["computation"]
        assert "epistemic" in computation
        assert "pragmatic" in computation

        # Policy precision parameters
        precision = cfe["policy_precision"]
        assert precision["beta"] == 2.0
        assert precision["adaptive"] is True
        assert precision["update_rule"] == "exponential_moving_average"


class TestProbabilityDistributionOperations:
    """Test parsing and evaluation of probability distribution operations."""

    def test_parse_categorical_distribution_operations(self):
        """Test parsing categorical distribution mathematical operations."""
        parser = GMNParser()

        spec = {
            "nodes": [
                {
                    "name": "categorical_belief",
                    "type": "belief",
                    "distribution_type": "categorical",
                    "operations": {
                        "normalization": {
                            "method": "softmax",
                            "formula": "exp(logits) / sum(exp(logits))",
                            "temperature": 1.0,
                        },
                        "entropy": {
                            "formula": "-sum(p * log(p))",
                            "regularization": 1e-8,
                        },
                        "update": {
                            "method": "bayesian",
                            "formula": "posterior ∝ likelihood * prior",
                            "normalization": "automatic",
                        },
                        "sampling": {
                            "method": "categorical_sample",
                            "temperature": 1.0,
                            "gumbel_noise": False,
                        },
                    },
                    "constraints": {
                        "simplex": True,
                        "non_negative": True,
                        "sum_to_one": True,
                    },
                }
            ],
            "edges": [],
        }

        result = parser.parse(spec)

        belief_node = result["nodes"][0]
        assert belief_node["distribution_type"] == "categorical"

        ops = belief_node["operations"]

        # Normalization operation
        norm = ops["normalization"]
        assert norm["method"] == "softmax"
        assert norm["formula"] == "exp(logits) / sum(exp(logits))"
        assert norm["temperature"] == 1.0

        # Entropy calculation
        entropy = ops["entropy"]
        assert entropy["formula"] == "-sum(p * log(p))"
        assert entropy["regularization"] == 1e-8

        # Bayesian update
        update = ops["update"]
        assert update["method"] == "bayesian"
        assert update["formula"] == "posterior ∝ likelihood * prior"

        # Constraints
        constraints = belief_node["constraints"]
        assert constraints["simplex"] is True
        assert constraints["sum_to_one"] is True

    def test_parse_dirichlet_distribution_operations(self):
        """Test parsing Dirichlet distribution mathematical operations."""
        parser = GMNParser()

        spec = {
            "nodes": [
                {
                    "name": "dirichlet_prior",
                    "type": "belief",
                    "distribution_type": "dirichlet",
                    "parameters": {
                        "alpha": [1.0, 1.0, 1.0, 1.0],
                        "concentration": 4.0,
                    },
                    "operations": {
                        "expectation": {
                            "formula": "alpha_i / sum(alpha)",
                            "variables": {
                                "alpha_i": "concentration_parameter_i",
                                "alpha": "concentration_vector",
                            },
                        },
                        "variance": {
                            "formula": "(alpha_i * (sum(alpha) - alpha_i)) / (sum(alpha)^2 * (sum(alpha) + 1))",
                            "variables": {
                                "alpha_i": "concentration_parameter_i"
                            },
                        },
                        "update": {
                            "method": "conjugate_bayesian",
                            "formula": "alpha_new = alpha_prior + counts",
                            "sufficient_statistics": "observation_counts",
                        },
                        "entropy": {
                            "formula": "log(B(alpha)) + (sum(alpha) - K) * psi(sum(alpha)) - sum((alpha_i - 1) * psi(alpha_i))",
                            "variables": {
                                "B": "beta_function",
                                "psi": "digamma_function",
                                "K": "dimension",
                            },
                        },
                    },
                }
            ],
            "edges": [],
        }

        result = parser.parse(spec)

        belief_node = result["nodes"][0]
        assert belief_node["distribution_type"] == "dirichlet"

        params = belief_node["parameters"]
        assert params["alpha"] == [1.0, 1.0, 1.0, 1.0]
        assert params["concentration"] == 4.0

        ops = belief_node["operations"]

        # Expectation formula
        expectation = ops["expectation"]
        assert expectation["formula"] == "alpha_i / sum(alpha)"

        # Variance formula
        variance = ops["variance"]
        assert "(sum(alpha)^2" in variance["formula"]

        # Conjugate Bayesian update
        update = ops["update"]
        assert update["method"] == "conjugate_bayesian"
        assert update["formula"] == "alpha_new = alpha_prior + counts"

    def test_parse_gaussian_distribution_operations(self):
        """Test parsing Gaussian distribution mathematical operations."""
        parser = GMNParser()

        spec = {
            "nodes": [
                {
                    "name": "gaussian_belief",
                    "type": "belief",
                    "distribution_type": "gaussian",
                    "parameters": {
                        "mean": 0.0,
                        "variance": 1.0,
                        "precision": 1.0,
                    },
                    "operations": {
                        "pdf": {
                            "formula": "(1/sqrt(2*pi*sigma^2)) * exp(-(x-mu)^2/(2*sigma^2))",
                            "variables": {
                                "mu": "mean",
                                "sigma": "standard_deviation",
                                "x": "observation",
                            },
                        },
                        "log_pdf": {
                            "formula": "-0.5 * log(2*pi*sigma^2) - (x-mu)^2/(2*sigma^2)",
                            "numerical_stability": True,
                        },
                        "update": {
                            "method": "precision_weighted",
                            "formula": {
                                "precision_new": "precision_prior + precision_likelihood",
                                "mean_new": "(precision_prior * mean_prior + precision_likelihood * observation) / precision_new",
                            },
                        },
                        "entropy": {
                            "formula": "0.5 * log(2*pi*e*sigma^2)",
                            "variables": {"e": "euler_constant"},
                        },
                    },
                }
            ],
            "edges": [],
        }

        result = parser.parse(spec)

        belief_node = result["nodes"][0]
        assert belief_node["distribution_type"] == "gaussian"

        params = belief_node["parameters"]
        assert params["mean"] == 0.0
        assert params["variance"] == 1.0
        assert params["precision"] == 1.0

        ops = belief_node["operations"]

        # PDF formula
        pdf = ops["pdf"]
        assert "exp(-(x-mu)^2/(2*sigma^2))" in pdf["formula"]

        # Log PDF with numerical stability
        log_pdf = ops["log_pdf"]
        assert log_pdf["numerical_stability"] is True

        # Precision-weighted update
        update = ops["update"]
        assert update["method"] == "precision_weighted"
        formulas = update["formula"]
        assert "precision_new" in formulas
        assert "mean_new" in formulas


class TestMatrixOperations:
    """Test parsing and evaluation of matrix operations in GMN specifications."""

    def test_parse_matrix_multiplication_expressions(self):
        """Test parsing matrix multiplication and linear algebra expressions."""
        parser = GMNParser()

        spec = {
            "nodes": [
                {
                    "name": "transition_dynamics",
                    "type": "transition",
                    "matrix_operations": {
                        "state_transition": {
                            "formula": "B @ s_t",
                            "variables": {
                                "B": "transition_matrix",
                                "s_t": "state_at_time_t",
                            },
                            "dimensions": {"B": [4, 4], "s_t": [4, 1]},
                        },
                        "observation_likelihood": {
                            "formula": "A @ s_t",
                            "variables": {
                                "A": "observation_matrix",
                                "s_t": "hidden_state",
                            },
                            "dimensions": {"A": [8, 4], "s_t": [4, 1]},
                        },
                        "policy_evaluation": {
                            "formula": "Q = R + gamma * B @ V",
                            "variables": {
                                "Q": "action_value_function",
                                "R": "reward_matrix",
                                "gamma": "discount_factor",
                                "B": "transition_matrix",
                                "V": "value_function",
                            },
                        },
                    },
                    "constraints": {
                        "stochastic_matrices": ["B"],
                        "column_normalization": True,
                        "non_negative": True,
                    },
                }
            ],
            "edges": [],
        }

        result = parser.parse(spec)

        transition_node = result["nodes"][0]
        ops = transition_node["matrix_operations"]

        # State transition
        state_trans = ops["state_transition"]
        assert state_trans["formula"] == "B @ s_t"
        dims = state_trans["dimensions"]
        assert dims["B"] == [4, 4]
        assert dims["s_t"] == [4, 1]

        # Observation likelihood
        obs_like = ops["observation_likelihood"]
        assert obs_like["formula"] == "A @ s_t"

        # Policy evaluation (Bellman equation)
        policy_eval = ops["policy_evaluation"]
        assert policy_eval["formula"] == "Q = R + gamma * B @ V"

        # Matrix constraints
        constraints = transition_node["constraints"]
        assert "B" in constraints["stochastic_matrices"]
        assert constraints["column_normalization"] is True

    def test_parse_eigenvalue_decomposition_expressions(self):
        """Test parsing eigenvalue decomposition and spectral analysis."""
        parser = GMNParser()

        spec = {
            "nodes": [
                {
                    "name": "spectral_analysis",
                    "type": "analysis",
                    "eigenvalue_decomposition": {
                        "matrix": "transition_matrix",
                        "decomposition": {
                            "formula": "B = Q * Lambda * Q^(-1)",
                            "variables": {
                                "B": "transition_matrix",
                                "Q": "eigenvector_matrix",
                                "Lambda": "eigenvalue_diagonal_matrix",
                            },
                        },
                        "stability_analysis": {
                            "spectral_radius": {
                                "formula": "max(abs(eigenvalues))",
                                "threshold": 1.0,
                                "stability_condition": "spectral_radius < 1",
                            },
                            "dominant_eigenvalue": {
                                "formula": "eigenvalue_with_largest_magnitude",
                                "convergence_rate": "abs(second_largest_eigenvalue)",
                            },
                        },
                        "stationary_distribution": {
                            "formula": "left_eigenvector_for_eigenvalue_1",
                            "normalization": "L1_norm",
                            "existence_condition": "eigenvalue_1_exists",
                        },
                    },
                }
            ],
            "edges": [],
        }

        result = parser.parse(spec)

        analysis_node = result["nodes"][0]
        eigen_decomp = analysis_node["eigenvalue_decomposition"]

        # Basic decomposition
        decomp = eigen_decomp["decomposition"]
        assert decomp["formula"] == "B = Q * Lambda * Q^(-1)"

        # Stability analysis
        stability = eigen_decomp["stability_analysis"]
        spectral_radius = stability["spectral_radius"]
        assert spectral_radius["formula"] == "max(abs(eigenvalues))"
        assert spectral_radius["threshold"] == 1.0

        # Stationary distribution
        stationary = eigen_decomp["stationary_distribution"]
        assert stationary["formula"] == "left_eigenvector_for_eigenvalue_1"
        assert stationary["normalization"] == "L1_norm"

    def test_parse_tensor_operations_expressions(self):
        """Test parsing tensor operations for multi-dimensional arrays."""
        parser = GMNParser()

        spec = {
            "nodes": [
                {
                    "name": "multi_factor_model",
                    "type": "state",
                    "tensor_operations": {
                        "joint_distribution": {
                            "formula": "kron(P_factor1, P_factor2, P_factor3)",
                            "variables": {
                                "P_factor1": "probability_factor_1",
                                "P_factor2": "probability_factor_2",
                                "P_factor3": "probability_factor_3",
                            },
                            "operation": "kronecker_product",
                        },
                        "marginal_distributions": {
                            "factor1_marginal": {
                                "formula": "sum(joint_distribution, axes=[1, 2])",
                                "axes": "non_factor1_axes",
                            },
                            "factor2_marginal": {
                                "formula": "sum(joint_distribution, axes=[0, 2])",
                                "axes": "non_factor2_axes",
                            },
                        },
                        "conditional_independence": {
                            "formula": "P(factor1, factor2 | factor3) = P(factor1 | factor3) * P(factor2 | factor3)",
                            "assumptions": [
                                "conditional_independence_given_factor3"
                            ],
                        },
                        "tensor_contraction": {
                            "formula": "einsum('ijk,jkl->il', tensor_a, tensor_b)",
                            "notation": "einstein_summation",
                            "variables": {
                                "tensor_a": "first_tensor",
                                "tensor_b": "second_tensor",
                            },
                        },
                    },
                }
            ],
            "edges": [],
        }

        result = parser.parse(spec)

        multi_factor_node = result["nodes"][0]
        tensor_ops = multi_factor_node["tensor_operations"]

        # Joint distribution via Kronecker product
        joint_dist = tensor_ops["joint_distribution"]
        assert joint_dist["formula"] == "kron(P_factor1, P_factor2, P_factor3)"
        assert joint_dist["operation"] == "kronecker_product"

        # Marginal distributions
        marginals = tensor_ops["marginal_distributions"]
        factor1_marg = marginals["factor1_marginal"]
        assert (
            "sum(joint_distribution, axes=[1, 2])" in factor1_marg["formula"]
        )

        # Conditional independence
        cond_indep = tensor_ops["conditional_independence"]
        assert (
            "P(factor1 | factor3) * P(factor2 | factor3)"
            in cond_indep["formula"]
        )

        # Tensor contraction
        contraction = tensor_ops["tensor_contraction"]
        assert contraction["notation"] == "einstein_summation"
        assert "einsum" in contraction["formula"]


class TestInformationTheoreticMeasures:
    """Test parsing information-theoretic measures and calculations."""

    def test_parse_mutual_information_expressions(self):
        """Test parsing mutual information calculations."""
        parser = GMNParser()

        spec = {
            "nodes": [
                {
                    "name": "information_measures",
                    "type": "analysis",
                    "mutual_information": {
                        "formula": "I(X; Y) = H(X) - H(X|Y)",
                        "variables": {
                            "X": "first_random_variable",
                            "Y": "second_random_variable",
                        },
                        "alternative_formulations": {
                            "kl_divergence": "KL(P(X,Y) || P(X)P(Y))",
                            "joint_entropy": "H(X) + H(Y) - H(X,Y)",
                            "conditional_entropy": "H(Y) - H(Y|X)",
                        },
                        "estimation": {
                            "method": "maximum_likelihood",
                            "bias_correction": "miller_madow",
                            "regularization": 1e-8,
                        },
                    },
                    "conditional_mutual_information": {
                        "formula": "I(X; Y | Z) = H(X|Z) - H(X|Y,Z)",
                        "variables": {"Z": "conditioning_variable"},
                    },
                    "transfer_entropy": {
                        "formula": "TE(X->Y) = I(Y_future; X_past | Y_past)",
                        "temporal_lag": 1,
                        "causality_direction": "X_to_Y",
                    },
                }
            ],
            "edges": [],
        }

        result = parser.parse(spec)

        info_node = result["nodes"][0]

        # Mutual information
        mi = info_node["mutual_information"]
        assert mi["formula"] == "I(X; Y) = H(X) - H(X|Y)"

        alt_forms = mi["alternative_formulations"]
        assert "KL(P(X,Y) || P(X)P(Y))" in alt_forms["kl_divergence"]
        assert "H(X) + H(Y) - H(X,Y)" in alt_forms["joint_entropy"]

        estimation = mi["estimation"]
        assert estimation["method"] == "maximum_likelihood"
        assert estimation["bias_correction"] == "miller_madow"

        # Conditional mutual information
        cmi = info_node["conditional_mutual_information"]
        assert cmi["formula"] == "I(X; Y | Z) = H(X|Z) - H(X|Y,Z)"

        # Transfer entropy
        te = info_node["transfer_entropy"]
        assert te["formula"] == "TE(X->Y) = I(Y_future; X_past | Y_past)"
        assert te["temporal_lag"] == 1

    def test_parse_entropy_rate_expressions(self):
        """Test parsing entropy rate calculations for stochastic processes."""
        parser = GMNParser()

        spec = {
            "nodes": [
                {
                    "name": "stochastic_process",
                    "type": "analysis",
                    "entropy_rate": {
                        "formula": "h = lim_{n->inf} H(X_n | X_{n-1}, ..., X_1) / n",
                        "variables": {"X_n": "sequence_element_at_time_n"},
                        "finite_approximation": {
                            "formula": "h_k = H(X_k | X_{k-1}, ..., X_{k-d+1})",
                            "memory_depth": "d",
                            "approximation_order": "k",
                        },
                        "markov_chain": {
                            "formula": "h = -sum(pi_i * sum(P_ij * log(P_ij)))",
                            "variables": {
                                "pi_i": "stationary_probability_i",
                                "P_ij": "transition_probability_i_to_j",
                            },
                            "convergence_condition": "ergodic_chain",
                        },
                    },
                    "excess_entropy": {
                        "formula": "E = sum_{k=1}^{inf} (H(X_k | X_{k-1}, ..., X_1) - h)",
                        "interpretation": "complexity_of_predictive_model",
                        "finite_approximation": {
                            "max_order": 10,
                            "convergence_threshold": 1e-6,
                        },
                    },
                }
            ],
            "edges": [],
        }

        result = parser.parse(spec)

        process_node = result["nodes"][0]

        # Entropy rate
        entropy_rate = process_node["entropy_rate"]
        assert "lim_{n->inf}" in entropy_rate["formula"]

        finite_approx = entropy_rate["finite_approximation"]
        assert "H(X_k | X_{k-1}" in finite_approx["formula"]

        markov_rate = entropy_rate["markov_chain"]
        assert "pi_i" in markov_rate["variables"]
        assert markov_rate["convergence_condition"] == "ergodic_chain"

        # Excess entropy
        excess_entropy = process_node["excess_entropy"]
        assert (
            excess_entropy["interpretation"]
            == "complexity_of_predictive_model"
        )
        finite_ee = excess_entropy["finite_approximation"]
        assert finite_ee["max_order"] == 10


class TestNumericalValidation:
    """Test numerical validation and evaluation of mathematical expressions."""

    def test_validate_mathematical_expression_syntax(self):
        """Test validation of mathematical expression syntax."""
        validator = GMNSchemaValidator()

        spec = {
            "nodes": [
                {
                    "name": "invalid_math",
                    "type": "belief",
                    "mathematical_expression": {
                        "formula": "log( / invalid_syntax",  # Invalid syntax
                        "variables": {"x": "input_variable"},
                    },
                }
            ],
            "edges": [],
        }

        is_valid, errors = validator.validate(spec)

        # Should detect syntax error in mathematical expression
        assert is_valid is False
        syntax_errors = [
            e
            for e in errors
            if "syntax" in e.lower() or "formula" in e.lower()
        ]
        assert len(syntax_errors) > 0

    def test_validate_dimensional_consistency_in_operations(self):
        """Test validation of dimensional consistency in mathematical operations."""
        validator = GMNSchemaValidator()

        spec = {
            "nodes": [
                {
                    "name": "dimension_mismatch",
                    "type": "operation",
                    "matrix_operation": {
                        "formula": "A @ B",
                        "variables": {"A": "matrix_a", "B": "matrix_b"},
                        "dimensions": {
                            "A": [3, 4],
                            "B": [5, 2],
                        },  # Incompatible for multiplication
                    },
                }
            ],
            "edges": [],
        }

        is_valid, errors = validator.validate(spec)

        # Should detect dimensional incompatibility
        assert is_valid is False
        dimension_errors = [e for e in errors if "dimension" in e.lower()]
        assert len(dimension_errors) > 0

    def test_validate_probability_constraints(self):
        """Test validation of probability constraints in expressions."""
        validator = GMNSchemaValidator()

        spec = {
            "nodes": [
                {
                    "name": "invalid_probability",
                    "type": "belief",
                    "probability_distribution": {
                        "values": [0.3, 0.5, 0.4],  # Sum > 1
                        "constraints": {"simplex": True, "sum_to_one": True},
                    },
                }
            ],
            "edges": [],
        }

        is_valid, errors = validator.validate(spec)

        # Should detect probability constraint violation
        assert is_valid is False
        prob_errors = [
            e
            for e in errors
            if "probability" in e.lower() or "sum" in e.lower()
        ]
        assert len(prob_errors) > 0
