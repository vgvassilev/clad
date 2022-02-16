---
title: "Project Meetings"
layout: gridlay
excerpt: "Project Meetings"
sitemap: false
permalink: /meetings/
---

# Project Meetings

{% assign sorted_meetings = site.data.meetinglist | sort: "date" | reverse %}
{% assign standing_meetings = site.data.standing_meetings %}
{% assign n_standing_meetings = standing_meetings | size %}

{% assign number_printed = 0 %}
{% for meeting in sorted_meetings %}

<div class="row">
<span id="{{meeting.label}}">&nbsp;</span>
<div class="col-sm-6 clearfix">
<div class="well" style="padding-left: 20px; padding-right: 20px">
  <a style="text-decoration:none;" href="#{{meeting.label}}">
    {{ meeting.name }} -- {{ meeting.date | date_to_long_string }} at {{meeting.time_cest}} Geneva (CH) Time
  </a>
<div>Connection information: {{meeting.connect}} <br />
</div><div>
  Agenda:
  <ul>{% for item in meeting.agenda %}
    <li><strong>{{item.title}}</strong>
      {% if item.speaker %}
        ({{item.speaker}})
      {% endif %}
      {% if item.slides %}
      <a style="text-decoration:none;" href="{{item.slides}}">Slides</a>
      {% endif %}
      {% if item.video %}
      <a style="text-decoration:none;" href="{{item.video}}">Video</a>
      {% endif %}
      {{ item.link }}
    </li>
    {% endfor %}</ul>
</div>
</div>
</div>

{% if number_printed < n_standing_meetings %}
{% assign smeeting = standing_meetings[number_printed] %}
<div class="col-sm-6 clearfix">
<div class="well" style="padding-left: 20px; padding-right: 20px">
  <a style="text-decoration:none;" href="#{{smeeting.label}}">
    {{ smeeting.name }} -- {{ smeeting.date }} at {{smeeting.time_cest}} Geneva (CH) Time
  </a>
<div>
  Connection information: {{smeeting.connect}} <br />
</div><div>
  Agenda:
  <ul>
    {% for item in smeeting.agenda %}
    <li><strong>{{item.title}}</strong>
      {% if item.speaker %}
        ({{item.speaker}}) [{{item.date|date: "%b %-d, %Y"}}]
      {% endif %}
      {{item.link | markdownify}}
    </li>
    {% endfor %}
   </ul>
</div>
</div>
</div>

{% endif %}

{% assign number_printed = number_printed | plus: 1 %}

</div>

{% endfor %} 
