"""Check what information enters the deployed monitor and its gradients."""
import pytest

torch = pytest.importorskip('torch')
from fusionbench.eval.context_student import ContextMLP, generate


def test_support_rms_and_query_isolation():
    support, query, _, prevalence = generate(42, [.05, .5, .95], 32, 6)
    torch.manual_seed(4)
    model = ContextMLP('joint', normalize=True)
    logits, context = model(support, query, prevalence)
    gain = torch.tensor([3., .25])
    scaled_logits, scaled_context = model(support * gain, query * gain, prevalence)
    torch.testing.assert_close(logits, scaled_logits)
    torch.testing.assert_close(context, scaled_context)
    _, another = model(support, query.flip(-1), prevalence)
    torch.testing.assert_close(context, another)


def test_observer_control_blocks_only_task_gradient():
    support, query, _, prevalence = generate(43, [.05, .5, .95], 32, 6)
    for mode in ('joint', 'observer'):
        torch.manual_seed(4)
        model = ContextMLP(mode)
        logits, _ = model(support, query, prevalence)
        logits.sum().backward()
        grad = model.encoder[0].weight.grad
        if mode == 'observer':
            assert grad is None
        else:
            assert grad.abs().max() > 0
        assert model.classifier[0].weight.grad.abs().max() > 0
