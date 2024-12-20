

def log_failures(reference_titles, generated_titles):
    """
    Identify and log failure modes in generated titles.

    Parameters:
    - reference_titles: list, actual titles by the author.
    - generated_titles: list, titles generated by the model.

    Returns:
    - List of failures.
    """
    failures = []
    for ref, gen in zip(reference_titles, generated_titles):
        if not gen or len(gen.split()) < 3 or 'undefined' in gen.lower():
            failures.append({'reference': ref, 'generated': gen, 'error': 'Too generic or invalid'})
    return failures






