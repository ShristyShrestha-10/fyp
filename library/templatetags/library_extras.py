from django import template
from django.utils.safestring import mark_safe

register = template.Library()

@register.filter
def similarity_color(value):
    """Return appropriate Bootstrap color class based on similarity score"""
    try:
        score = float(value)
        if score < 0.5:
            return 'danger'
        elif score < 0.7:
            return 'warning'
        else:
            return 'success'
    except (ValueError, TypeError):
        return 'secondary'

@register.filter
def similarity_progress(value):
    """Create a progress bar for similarity score"""
    try:
        score = float(value)
        percentage = score * 100
        color_class = similarity_color(score)
        
        html = f"""
        <div class="progress" style="height: 15px;">
            <div class="progress-bar bg-{color_class}" 
                 role="progressbar" 
                 style="width: {percentage:.1f}%;" 
                 aria-valuenow="{percentage:.1f}" 
                 aria-valuemin="0" 
                 aria-valuemax="100">
                {score:.2f}
            </div>
        </div>
        """
        return mark_safe(html)
    except (ValueError, TypeError):
        return 'N/A'